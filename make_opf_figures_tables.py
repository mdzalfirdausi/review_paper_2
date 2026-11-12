#!/usr/bin/env python3
"""Create Hira-style OPF bibliometric figures and LaTeX tables from builder v3.
Run: python make_opf_figures_tables.py --root .
Dependencies: pandas numpy matplotlib. No model fitting or source-file changes.
All denominators refer to the supplied DOI-linked metadata, not all raw exports.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, PURPLE = '#28648c', '#bf7522', '#765393'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                    '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def formatted(value):
    if pd.isna(value):
        return 'NR'
    if isinstance(value, (int, np.integer)):
        return f'{value:,}'
    if isinstance(value, (float, np.floating)):
        return f'{value:,.0f}' if float(value).is_integer() else f'{value:,.3f}'
    return str(value)


def table(path, caption, label, frame, note, columns=None):
    columns = columns or ('X' + 'r' * (len(frame.columns) - 1))
    lines = [r'\begin{table}[htbp]', r'\centering', r'\caption{' + caption + '}',
             r'\label{' + label + '}', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
             r'\renewcommand{\arraystretch}{1.15}',
             r'\begin{tabularx}{\textwidth}{' + columns + '}', r'\toprule',
             ' & '.join(r'\shortstack{' + r'\\'.join(escape(w) for w in c.split()) + '}'
                        for c in frame.columns) + r' \\', r'\midrule']
    for row in frame.itertuples(index=False, name=None):
        lines.append(' & '.join(escape(formatted(v)) for v in row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabularx}', r'\vspace{2mm}',
                  r'\begin{minipage}{\textwidth}', r'\footnotesize',
                  r'\textit{Notes:} ' + note, r'\end{minipage}', r'\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def barh(ax, series, title, color=BLUE):
    series = series.sort_values(ascending=False).head(10).iloc[::-1]
    if series.empty:
        ax.text(.5, .5, 'No usable records', ha='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    labels = ['\n'.join(textwrap.wrap(str(x), 36)) for x in series.index]
    ax.barh(range(len(series)), series.values, color=color)
    ax.set_yticks(range(len(series)), labels)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--input', help='Optional publication metadata CSV path')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(args.input).resolve() if args.input else root / 'output/opf_metadata/opf_publication_metadata.csv'
    out = root / 'output/opf_hira_results'
    figs, tabs, data = [out / name for name in ['figures', 'tables', 'data']]
    for folder in [figs, tabs, data]:
        folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = ['doi_norm', 'Title', 'Year', 'Source title', 'Scopus_citations',
                'in_lens', 'in_overton', 'patent_citation_count', 'policy_citation_count']
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError(f'Missing input columns: {sorted(absent)}')
    if df.doi_norm.eq('').any() or df.doi_norm.duplicated().any():
        raise ValueError('Expected unique nonempty normalized DOIs.')
    for column in ['in_lens', 'in_overton']:
        raw = df[column].str.lower().str.strip()
        if not raw.isin(['true', 'false', '1', '0']).all():
            raise ValueError(f'Invalid boolean values in {column}')
        df[column] = raw.isin(['true', '1'])
    for column in ['Year', 'Scopus_citations', 'patent_citation_count', 'policy_citation_count']:
        raw = df[column]
        df[column] = pd.to_numeric(raw, errors='coerce')
        if (raw.ne('') & df[column].isna()).any():
            raise ValueError(f'Invalid numeric values in {column}')
        if df[column].dropna().lt(0).any():
            raise ValueError(f'Negative values in {column}')
    if df.Year.isna().any() or df.Year.mod(1).ne(0).any():
        raise ValueError('Missing/noninteger publication years must be resolved first.')
    df['Year'] = df.Year.astype(int)
    if df.Scopus_citations.isna().any():
        raise ValueError('Missing scholarly citation counts: resolve before generating scholarly summaries.')
    n = len(df)
    if not n:
        raise ValueError('Input population is empty.')
    df['Source title'] = df['Source title'].replace('', 'Unknown source')
    fragments = []

    def figure(name, fig, caption):
        fig.savefig(figs / f'{name}.pdf', bbox_inches='tight', facecolor='white')
        fig.savefig(figs / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        fragment = '\n'.join([r'\begin{figure}[htbp]', r'\centering',
            r'\includegraphics[width=\linewidth]{output/opf_hira_results/figures/' + name + '.pdf}',
            r'\caption{' + caption + '}', r'\label{fig:' + name + '}', r'\end{figure}'])
        fragments.append(fragment)
        print('Figure:', name)

    def save_table(name, frame, caption, note, columns=None):
        frame.to_csv(data / f'{name}.csv', index=False, encoding='utf-8-sig')
        table(tabs / f'{name}.tex', caption, 'tab:' + name, frame, note, columns)
        print('Table:', name)

    # Academic annual patterns: citations are snapshot totals grouped by publication year.
    annual = df.groupby('Year').agg(Publications=('doi_norm', 'size'),
                                    Citations=('Scopus_citations', 'sum'))
    annual['Mean citations'] = annual.Citations / annual.Publications
    annual.to_csv(data / 'academic_annual.csv', encoding='utf-8-sig')
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout='constrained')
    for ax, col in zip(axes, annual.columns):
        ax.plot(annual.index, annual[col], marker='o', ms=3, color=BLUE)
        ax.set_ylabel(col); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Publication year')
    figure('opf_annual_research', fig, 'Annual publications, accumulated Scopus citations and mean citations in the DOI-linked ML--OPF population. Citations are snapshot totals grouped by publication year, not citations received during each year. The latest publication year may be incomplete.')

    journals = df.groupby('Source title').agg(Publications=('doi_norm', 'size'),
                                             Citations=('Scopus_citations', 'sum'))
    journals['Mean citations'] = journals.Citations / journals.Publications
    journals.to_csv(data / 'academic_journals_all.csv', encoding='utf-8-sig')
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), layout='constrained')
    for ax, metric in zip(axes, journals.columns):
        barh(ax, journals[metric], metric)
    figure('opf_leading_journals', fig, 'Leading publication sources ranked separately by publication count, accumulated Scopus citations and mean citations per publication. Mean-citation rankings impose no minimum publication count; sources with few papers can rank highly.')
    save_table('opf_leading_academic_sources', journals.nlargest(10, 'Publications').reset_index(),
               'Leading sources by publication volume in the DOI-linked ML--OPF population.',
               'Citations refer to the Scopus export snapshot. Source names are retained as exported.')

    # Academic citation distribution.
    bins = [-.5, .5, 9.5, 49.5, 99.5, 499.5, float('inf')]
    labels = ['0', '1--9', '10--49', '50--99', '100--499', '500+']
    hist = pd.cut(df.Scopus_citations, bins=bins, labels=labels).value_counts(sort=False)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.bar(labels, hist.values, color=BLUE)
    ax.set_xlabel('Scopus citations per publication'); ax.set_ylabel('Publications')
    figure('opf_citation_distribution', fig, 'Distribution of accumulated scholarly citations across the DOI-linked ML--OPF population.')
    hist.rename('Publications').to_csv(data / 'academic_citation_distribution.csv', encoding='utf-8-sig')

    indicators = []
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout='constrained')
    for name, flag, count, color in [('Patent', 'in_lens', 'patent_citation_count', ORANGE),
                                     ('Policy', 'in_overton', 'policy_citation_count', PURPLE)]:
        part = df.loc[df[flag]].copy()
        total = float(part[count].sum()) if part[count].notna().all() else np.nan
        indicators.append({'Pathway': name, 'Population': n, 'Cited publications': len(part),
                           'Coverage (%)': 100 * len(part) / n, 'External citations': total,
                           'Density': total / n,
                           'Intensity': total / len(part) if len(part) else np.nan})
        year = part.groupby('Year').agg(Publications=('doi_norm', 'size'),
                                       **{'External citations': (count, lambda x: x.sum() if x.notna().all() else np.nan)})
        year = year.reindex(range(df.Year.min(), df.Year.max() + 1))
        observed = set(part.Year)
        for y in year.index:
            if y not in observed:
                year.loc[y] = 0
        year.to_csv(data / f'{name.lower()}_annual.csv', encoding='utf-8-sig')
        axes[0].plot(year.index, year.Publications, label=name, color=color)
        axes[1].plot(year.index, year['External citations'], label=name, color=color)
        sources = part.groupby('Source title').agg(
            **{'Cited publications': ('doi_norm', 'size'),
               'External citations': (count, lambda x: x.sum() if x.notna().all() else np.nan)})
        sources['Population publications'] = journals.Publications.reindex(sources.index)
        sources['Within-source coverage (%)'] = 100 * sources['Cited publications'] / sources['Population publications']
        sources['Intensity'] = sources['External citations'] / sources['Cited publications']
        save_table(f'opf_{name.lower()}_sources', sources.nlargest(10, 'Cited publications').reset_index(),
                   f'Leading sources of {name.lower()}-cited ML--OPF publications.',
                   'Coverage uses all DOI-linked Scopus publications in the corresponding source as its denominator. External citation totals are retained from the supplied export; NR denotes unavailable counts.')
        top = part.sort_values([count, 'Scopus_citations'], ascending=False).head(10)
        display = top[['Title', 'Year', 'Scopus_citations', count]].copy()
        display.columns = ['Publication title', 'Year', 'Scopus citations', f'{name} citations']
        save_table(f'opf_top_{name.lower()}_cited', display,
                   f'Ten most {name.lower()}-cited publications in the DOI-linked ML--OPF population.',
                   'Ranked by external citation count, with Scopus citations breaking ties. NR denotes an unavailable count.', 'Xrrr')
    for ax in axes:
        ax.legend(frameon=False); ax.grid(alpha=.2)
    axes[0].set_ylabel('Cited publications'); axes[1].set_ylabel('External citations')
    axes[-1].set_xlabel('Publication year of cited research')
    figure('opf_external_citation_trends', fig, 'Patent- and policy-cited publications and their external citation totals, grouped by research publication year. These are not annual patent or policy citation-event counts; the exports have different observation dates.')
    save_table('opf_citation_indicators', pd.DataFrame(indicators),
               'Patent and policy citation indicators for the DOI-linked ML--OPF population.',
               'Coverage is the percentage cited at least once; density is external citations divided by the full DOI-linked population; intensity is external citations divided by cited publications. Patent counts refer to the Lens reported patent-citation measure, not patent families. NR indicates an incomplete citation total.')
    top = df.sort_values('Scopus_citations', ascending=False).head(10)
    display = top[['Title', 'Year', 'Scopus_citations', 'patent_citation_count', 'policy_citation_count']].copy()
    display.columns = ['Publication title', 'Year', 'Scopus citations', 'Patent citations', 'Policy citations']
    save_table('opf_top_academic_cited', display, 'Ten most academically cited publications in the DOI-linked ML--OPF population.',
               'Scholarly, patent and policy citation counts come from their respective export snapshots. Zero indicates no matching citation in the supplied exports; NR denotes an unknown count.', 'Xrrrr')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, count, name, color in zip(axes, ['patent_citation_count', 'policy_citation_count'], ['Patent', 'Policy'], [ORANGE, PURPLE]):
        valid = df[count].notna()
        ax.scatter(df.loc[valid, 'Scopus_citations'], df.loc[valid, count], s=14, alpha=.45, color=color, edgecolors='none')
        ax.set_xscale('symlog', linthresh=1); ax.set_yscale('symlog', linthresh=1)
        ax.set_xlabel('Scopus citations'); ax.set_ylabel(name + ' citations')
        ax.set_title(name + ' versus scholarly citations'); ax.grid(alpha=.2)
    figure('opf_academic_external_citations', fig, 'Scholarly citations compared with patent and policy citations. Symmetric logarithmic axes retain zero values. Citation counts measure documented attention rather than implementation or causal impact.')

    # Author counts: associate semicolon-separated display names with stable IDs.
    authors, names, skipped = Counter(), {}, 0
    if {'Authors', 'Author_IDs'}.issubset(df.columns):
        for row in df[['Authors', 'Author_IDs']].itertuples(index=False, name=None):
            ns = [x.strip() for x in row[0].split(';') if x.strip()]
            ids = [x.strip() for x in row[1].split(';') if x.strip()]
            if len(ns) != len(ids) or not ids:
                skipped += 1; continue
            authors.update(set(ids))
            names.update(zip(ids, ns))
    countries = Counter()
    if 'Affiliation_countries' in df:
        for value in df.Affiliation_countries:
            countries.update(set(x.strip() for x in value.split(';') if x.strip()))
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), layout='constrained')
    a = pd.Series({f'{names[k]} [{k}]': v for k, v in authors.most_common(10)}, dtype=float)
    barh(axes[0], a, 'Productive authors (full counting)')
    barh(axes[1], pd.Series(dict(countries), dtype=float), 'Affiliation countries (full counting)')
    figure('opf_authors_countries', fig, f'Productive authors identified by Scopus author ID and countries extracted from affiliation addresses. Each publication contributes once to each associated author or country. This is not first-author-country counting. Author parsing omitted {skipped} records; unparsed country segments remain excluded.')
    pd.DataFrame([{'Author ID': k, 'Author': names[k], 'Publications': v} for k, v in authors.most_common()]).to_csv(data / 'authors_full_count.csv', index=False, encoding='utf-8-sig')
    pd.Series(dict(countries), name='Publications').rename_axis('Country').to_csv(data / 'countries_full_count.csv', encoding='utf-8-sig')

    (out / 'figure_insertions.tex').write_text('\n\n'.join(fragments) + '\n', encoding='utf-8')
    lines = [r'% Requires graphicx, booktabs and tabularx. Paths assume the project root.',
             r'% Insert within your existing Results and Discussion section.',
             r'\subsection{Overview of ML--OPF Research}\label{subsec:opf_research_overview}',
             f'The DOI-linked analysis population comprised {n:,} unique publications, including {int(df.in_lens.sum()):,} patent-cited and {int(df.in_overton.sum()):,} policy-cited publications. These counts describe the exported metadata population and are distinct from the final topic-modelling population. Publications without a usable DOI are outside this bibliometric population.',
             fragments[0], fragments[1],
             r'\input{output/opf_hira_results/tables/opf_leading_academic_sources.tex}',
             fragments[-1], fragments[2],
             r'\input{output/opf_hira_results/tables/opf_top_academic_cited.tex}',
             r'\subsection{Patent and Policy Citation Patterns}\label{subsec:opf_external_citations}',
             r'\input{output/opf_hira_results/tables/opf_citation_indicators.tex}',
             fragments[3],
             r'\input{output/opf_hira_results/tables/opf_patent_sources.tex}',
             r'\input{output/opf_hira_results/tables/opf_policy_sources.tex}',
             r'\input{output/opf_hira_results/tables/opf_top_patent_cited.tex}',
             r'\input{output/opf_hira_results/tables/opf_top_policy_cited.tex}', fragments[4],
             r'% Topic modelling and research-gap subsections require final topic assignments and representative papers.']
    (out / 'results_bibliometrics.tex').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    summary = {'population': n, 'basis': 'DOI-linked metadata, not all raw Scopus records',
               'lens': int(df.in_lens.sum()), 'overton': int(df.in_overton.sum()),
               'figures': len(fragments), 'tables': len(list(tabs.glob('*.tex'))),
               'author_records_not_parsed': skipped,
               'missing_affiliation_country_records': int(df.get('Affiliation_countries', pd.Series('', index=df.index)).eq('').sum())}
    (out / 'generation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'README.txt').write_text(
        'Generated from: ' + str(source) + '\nPopulation: DOI-linked records only.\n'
        'All annual citation totals are grouped by the year of the cited paper, not citing-document year.\n'
        'Countries use all parsed affiliations, not first-author country.\n'
        'Tables use the source export citation measures; counts do not establish causal impact.\n'
        'Raw papers without valid DOIs and the final topic-modelled population have different denominators.\n'
        'Insert results_bibliometrics.tex from the project root; requires booktabs, tabularx, graphicx.\n'
        'No topic figures are generated without final topic assignments.\n', encoding='utf-8')
    archive = root / 'output/opf_hira_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out.parent))
    print(f'\nDONE: {len(fragments)} figures (PDF + PNG), {summary["tables"]} LaTeX tables.')
    print(f'LaTeX: {out / "results_bibliometrics.tex"}\nArchive: {archive}')
    print(f'Population: {n:,}; Lens: {summary["lens"]:,}; Overton: {summary["overton"]:,}')


if __name__ == '__main__':
    main()
