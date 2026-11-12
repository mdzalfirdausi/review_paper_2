#!/usr/bin/env python3
"""Create Hira-style OPF bibliometric figures and LaTeX tables from builder v3.
Run: python make_opf_figures_tables_v2.py --root .
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


# Compact plotting helpers. Statistics are recomputed from the original CSV.
TEAL = '#168778'
INK = '#243746'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.edgecolor': '#b8c2ca', 'axes.linewidth': .65,
                     'grid.color': '#dce2e7', 'grid.linewidth': .5,
                     'savefig.pad_inches': .035})


def tidy(ax, grid='y'):
    ax.grid(axis=grid, alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=3)


def year_ticks(ax, years):
    lo, hi = int(min(years)), int(max(years))
    step = 5 if hi - lo > 20 else 3 if hi - lo > 12 else 2
    ax.set_xticks(list(range(((lo + step - 1) // step) * step, hi + 1, step)))
    ax.set_xlim(lo - .65, hi + .65)


def compact_annual(annual):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(7.4, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=.48)
    ax, mean = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.bar(annual.index, annual.Publications, width=.72, color=BLUE, alpha=.8)
    citations = ax.twinx()
    citations.plot(annual.index, annual.Citations, color=PURPLE, lw=1.45, marker='o', ms=2)
    ax.set_ylabel('Publications', color=BLUE)
    citations.set_ylabel('Scopus citations', color=PURPLE)
    citations.set_ylim(bottom=0)
    ax.tick_params(axis='y', colors=BLUE); citations.tick_params(axis='y', colors=PURPLE)
    citations.spines['right'].set_visible(True)
    citations.spines['right'].set_color(PURPLE)
    mean.plot(annual.index, annual['Mean citations'], color=TEAL, lw=1.4)
    mean.fill_between(annual.index, annual['Mean citations'], color=TEAL, alpha=.1)
    mean.set_ylabel('Citations per publication')
    ax.set_title('(a) Research volume and scholarly attention', loc='left')
    mean.set_title('(b) Citation efficiency', loc='left')
    for a in [ax, mean]:
        a.set_xlabel('Publication year'); tidy(a); year_ticks(a, annual.index)
        a.set_ylim(bottom=0)
        if max(annual.index) >= 2026:
            a.axvspan(2025.5, max(annual.index) + .65, color='#81909d', alpha=.1)
    from matplotlib.ticker import MaxNLocator
    mean.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    fig.legend(handles=[Patch(facecolor=BLUE, alpha=.8, label='Publications'),
                        Line2D([], [], color=PURPLE, label='Scopus citations'),
                        Line2D([], [], color=TEAL, label='Mean citations')],
               loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False,
               handlelength=1.5, columnspacing=1.8)
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.82)
    return fig


def compact_journals(journals):
    # Equal-weight mean percentile rank; do not mix raw metrics with different units.
    ranked = journals.copy()
    ranked['Combined score'] = ranked[['Publications', 'Citations', 'Mean citations']].rank(pct=True).mean(axis=1) * 100
    keep = set()
    for c in ranked.columns:
        keep.update(ranked.nlargest(10, c).index)
    shown = ranked.loc[ranked.index.isin(keep)].sort_values('Combined score', ascending=False)
    abbreviations = {
        'IEEE Transactions on Power Systems': 'IEEE Trans. Power Systems',
        'IEEE Transactions on Smart Grid': 'IEEE Trans. Smart Grid',
        'IEEE Transactions on Evolutionary Computation': 'IEEE Trans. Evolutionary Computation',
        'International Journal of Electrical Power and Energy Systems': 'Int. J. Electrical Power & Energy Systems',
        'Renewable and Sustainable Energy Reviews': 'Renewable & Sustainable Energy Reviews',
        'Engineering Applications of Artificial Intelligence': 'Engineering Applications of AI',
    }
    labels = ['\n'.join(textwrap.wrap(abbreviations.get(s, s), 44)) for s in shown.index]
    height = max(3.8, sum(max(1, x.count('\n') + 1) for x in labels) * .14 + .85)
    fig, ax = plt.subplots(figsize=(7.4, height))
    fig.subplots_adjust(left=.47, right=.985, top=1 - .48 / height, bottom=.25 / height)
    positions = [0, 1.3, 2.6, 3.9]
    colors = [BLUE, PURPLE, TEAL, ORANGE]
    row_heights = np.array([max(1, label.count('\n') + 1) for label in labels], dtype=float)
    edges = np.concatenate([[0], np.cumsum(row_heights)])
    y = (edges[:-1] + edges[1:]) / 2
    for pos, col, color in zip(positions, shown.columns, colors):
        vals = shown[col].to_numpy(float)
        sizes = 9 + 110 * vals / max(vals.max(), 1)
        ax.scatter(np.full(len(y), pos), y, s=sizes, c=color, alpha=.8, edgecolors='white', linewidths=.3)
        for yi, val in zip(y, vals):
            label = f'{val:,.0f}' if col in ['Publications', 'Citations'] else f'{val:,.1f}'
            ax.text(pos + .21, yi, label, va='center', fontsize=6.8, color=INK)
    for i in range(0, len(y), 2):
        ax.axhspan(edges[i], edges[i+1], color='#edf2f5', zorder=-1)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xticks(positions, ['Publications', 'Citations', 'Mean\ncitations', 'Combined\nscore'])
    ax.xaxis.tick_top(); ax.tick_params(axis='x', length=0, pad=9)
    ax.tick_params(axis='y', length=0, pad=8, labelsize=6.8)
    ax.set_xlim(-.25, 4.75); ax.set_ylim(edges[-1], 0)
    for spine in ax.spines.values(): spine.set_visible(False)
    return fig, ranked


def compact_distribution(df, hist):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={'width_ratios': [1.55, 1]})
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.84, wspace=.42)
    share = 100 * hist.to_numpy() / len(df)
    colors = ['#b8c8d5', BLUE, '#3d7f9d', '#5d9eac', '#91bdc2', '#c5dedc']
    axes[0].bar(np.arange(len(hist)), share, width=.66, color=colors)
    for i, (pct, count) in enumerate(zip(share, hist)):
        axes[0].text(i, pct + max(share)*.025, f'{pct:.1f}%\n(n={count:,})', ha='center', va='bottom', fontsize=6.3)
    axes[0].set_xticks(range(len(hist)), ['0', '1–9', '10–49', '50–99', '100–499', '500+'])
    axes[0].set_ylim(0, max(share) * 1.3)
    axes[0].set_ylabel('Publications (%)'); axes[0].set_xlabel('Scopus citations')
    vals = np.sort(df.Scopus_citations.to_numpy(float))
    axes[1].step(vals, 100*(len(vals)-np.arange(len(vals)))/len(vals), where='post', color=PURPLE, lw=1.4)
    axes[1].set_xscale('symlog', linthresh=1)
    axes[1].set_xlabel('Scopus citations'); axes[1].set_ylabel('Publications with ≥ x citations (%)')
    axes[1].set_ylim(0, 102)
    axes[0].set_title('(a) Citation distribution', loc='left'); axes[1].set_title('(b) Citation concentration', loc='left')
    for ax in axes: tidy(ax)
    return fig


def compact_external(df):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    fig.subplots_adjust(left=.07, right=.93, bottom=.2, top=.77, wspace=.5)
    years = np.arange(df.Year.min(), df.Year.max() + 1)
    maxpapers = max(df.loc[df.in_lens].groupby('Year').size().max() if df.in_lens.any() else 0,
                    df.loc[df.in_overton].groupby('Year').size().max() if df.in_overton.any() else 0, 1)
    maxcites = max(df.groupby('Year').patent_citation_count.sum().max(), df.groupby('Year').policy_citation_count.sum().max(), 1)
    for ax, flag, count, label, color in zip(axes, ['in_lens', 'in_overton'], ['patent_citation_count', 'policy_citation_count'], ['(a) Patent-cited research', '(b) Policy-cited research'], [ORANGE, PURPLE]):
        part = df.loc[df[flag]]
        pubs = part.groupby('Year').size().reindex(years, fill_value=0)
        cites = part.groupby('Year')[count].agg(lambda x: x.sum() if x.notna().all() else np.nan).reindex(years, fill_value=0)
        ax.bar(years, pubs, width=.72, color=color, alpha=.65)
        twin = ax.twinx(); twin.plot(years, cites, color=color, lw=1.6, marker='o', ms=2)
        twin.spines['right'].set_visible(True); twin.spines['right'].set_color(color)
        twin.tick_params(axis='y', colors=color); twin.set_ylabel('External citations', color=color)
        ax.set_ylim(0, maxpapers*1.18); twin.set_ylim(0, maxcites*1.18)
        ax.set_ylabel('Cited publications'); ax.set_xlabel('Research publication year')
        ax.set_title(label, loc='left', pad=22)
        total = part[count].sum() if part[count].notna().all() else np.nan
        detail = f'n={len(part):,}  |  coverage={100*len(part)/len(df):.2f}%'
        if len(part) and np.isfinite(total): detail += f'  |  intensity={total/len(part):.2f}'
        ax.text(0, 1.045, detail, transform=ax.transAxes, fontsize=6.6, color=INK)
        year_ticks(ax, years); tidy(ax)
        if max(years) >= 2026: ax.axvspan(2025.5, max(years)+.65, color='#81909d', alpha=.1)
    fig.legend(handles=[Patch(facecolor='#6c7d8a', alpha=.65, label='Cited publications'),
                        Line2D([], [], color='#6c7d8a', marker='o', ms=3, label='External citations')],
               loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    return fig


def mark_points(ax, points):
    # Greedy annotation placement in display coordinates; no extra dependencies.
    from matplotlib.text import Text
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    offsets = [(7,7)] + [(dx,dy) for r in [14,24,36,50,70,90] for dx,dy in [(r,7),(7,r),(-r,7),(7,-r),(r,r),(-r,r),(r,-r),(-r,-r)]]
    bounds = ax.get_window_extent(renderer).padded(-1)
    for x, y, label in points:
        chosen = None
        for dx, dy in offsets:
            trial = ax.annotate(label, (x,y), xytext=(dx,dy), textcoords='offset points',
                                fontsize=5.5, color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.25),
                                arrowprops=dict(arrowstyle='-', color='#aab3bc', lw=.35))
            trial.update_positions(renderer)
            box = Text.get_window_extent(trial, renderer).padded(2)
            penalty = sum(box.overlaps(b) for b in occupied) + 10 * (not bounds.contains(box.x0, box.y0) or not bounds.contains(box.x1, box.y1))
            trial.remove()
            if chosen is None or penalty < chosen[0]: chosen = (penalty,dx,dy)
            if penalty == 0: break
        _, dx, dy = chosen
        annotation = ax.annotate(label, (x,y), xytext=(dx,dy), textcoords='offset points', fontsize=5.5,
                                 color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.25),
                                 arrowprops=dict(arrowstyle='-', color='#aab3bc', lw=.35))
        annotation.update_positions(renderer)
        occupied.append(Text.get_window_extent(annotation, renderer).padded(2))


def compact_citations(df, data):
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1,2,figsize=(7.4,3.25))
    fig.subplots_adjust(left=.075,right=.985,bottom=.2,top=.83,wspace=.3)
    for ax, count, name, color in zip(axes, ['patent_citation_count','policy_citation_count'], ['Patent','Policy'], [ORANGE,PURPLE]):
        valid = df.loc[df[count].notna()].copy()
        x=np.log10(1+valid.Scopus_citations); y=np.log10(1+valid[count])
        ax.hexbin(x,y,gridsize=(45,25),mincnt=1,cmap='Greys',norm=LogNorm(),linewidths=0,alpha=.7)
        positive=valid[count].gt(0)
        ax.scatter(x[positive],y[positive],s=8,color=color,alpha=.5,edgecolors='none')
        academic=valid.nlargest(10,'Scopus_citations'); external=valid.nlargest(10,count)
        xmax=max(1,float(x.max())); ymax=max(.5,float(y.max()))
        ax.set_xlim(-.06,xmax*1.08+.1); ax.set_ylim(-.04,ymax*1.18+.06)
        points=[]; records=[]
        for frame, prefix, marker, col in [(academic,'A','o',BLUE),(external,'E','*',color)]:
            for rank, (_, row) in enumerate(frame.iterrows(),1):
                xx,yy=np.log10(1+row.Scopus_citations),np.log10(1+row[count])
                ax.scatter([xx],[yy],s=28 if marker=='o' else 45,marker=marker,
                           facecolors='none' if marker=='o' else col,edgecolors=col,linewidths=.8,zorder=5)
                points.append((xx,yy,prefix+str(rank)))
                records.append({'Plot label':prefix+str(rank),'doi_norm':row.doi_norm,'Title':row.Title,
                                'Scopus citations':row.Scopus_citations,'External citations':row[count]})
        pd.DataFrame(records).to_csv(data/f'{name.lower()}_scatter_key.csv',index=False,encoding='utf-8-sig')
        for axis, maximum, setter in [('x',valid.Scopus_citations.max(),ax.set_xticks),('y',valid[count].max(),ax.set_yticks)]:
            ticks=[0,1]+[10**k for k in range(1, int(np.ceil(np.log10(max(maximum,1))))+1)]
            ticks=[v for v in ticks if v <= max(maximum,1)*1.3]
            setter(np.log10(1+np.array(ticks)),[f'{v:,}' for v in ticks])
        ax.set_xlabel('Scopus citations (log1p scale)');ax.set_ylabel(name+' citations (log1p scale)')
        ax.set_title(name+' versus academic attention',loc='left')
        tidy(ax,grid='both');mark_points(ax,points)
    fig.text(.5,.97,'A1–A10: most academically cited    •    E1–E10: most externally cited',ha='center',va='top',fontsize=7)
    return fig


def compact_people(authors,names,countries):
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.3))
    fig.subplots_adjust(left=.13,right=.95,bottom=.17,top=.84,wspace=1.0)
    sets=[[(names[k],v) for k,v in authors.most_common(10)], countries.most_common(20)]
    short={'Iran, Islamic Republic of':'Iran','Korea, Republic of':'South Korea','Taiwan, Province of China':'Taiwan','Russian Federation':'Russia'}
    for ax,items,title,col in zip(axes,sets,['(a) Productive authors','(b) Affiliation countries'],[BLUE,TEAL]):
        if not items:
            ax.text(.5,.5,'No usable records',transform=ax.transAxes,ha='center');continue
        items=items[::-1]; labels=[short.get(k,k) for k,v in items]; vals=[v for k,v in items]
        ax.hlines(range(len(vals)),0,vals,color=col,alpha=.25,lw=2)
        ax.scatter(vals,range(len(vals)),s=27,color=col,zorder=3)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.035,i,f'{v:,}',va='center',fontsize=7)
        ax.set_yticks(range(len(vals)),labels);ax.tick_params(axis='y',length=0,labelsize=7)
        ax.set_xlim(0,max(vals)*1.25);ax.set_ylim(-.6,len(vals)-.4)
        ax.set_xlabel('Publications');ax.set_title(title,loc='left',pad=10);tidy(ax,'x')
    fig.text(.5,.98,'Full counting; author IDs retained in the supporting CSV',ha='center',va='top',fontsize=7,color=INK)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--input', help='Optional publication metadata CSV path')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = Path(args.input).resolve() if args.input else root / 'output/opf_metadata/opf_publication_metadata.csv'
    out = root / 'output/opf_hira_results_v2'
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
    future = df.loc[df.Year.gt(2026)]
    future.to_csv(data / 'future_dated_records.csv', index=False, encoding='utf-8-sig')
    if len(future): print(f'NOTE: {len(future)} records have years after 2026; retained and shaded in temporal plots.')
    if not n:
        raise ValueError('Input population is empty.')
    df['Source title'] = df['Source title'].replace('', 'Unknown source')
    fragments = []

    def figure(name, fig, caption):
        fig.savefig(figs / f'{name}.pdf', bbox_inches='tight', facecolor='white')
        fig.savefig(figs / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        fragment = '\n'.join([r'\begin{figure}[htbp]', r'\centering',
            r'\includegraphics[width=\linewidth]{output/opf_hira_results_v2/figures/' + name + '.pdf}',
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
    fig = compact_annual(annual)
    figure('opf_annual_research', fig, 'Annual publications, accumulated Scopus citations and mean citations in the DOI-linked ML--OPF population. Citations are snapshot totals grouped by publication year, not citations received during each year. The shaded area identifies 2026 and later records, including incomplete or future-dated publication years; these records are retained as exported.')

    journals = df.groupby('Source title').agg(Publications=('doi_norm', 'size'),
                                             Citations=('Scopus_citations', 'sum'))
    journals['Mean citations'] = journals.Citations / journals.Publications
    journals.to_csv(data / 'academic_journals_all.csv', encoding='utf-8-sig')
    fig, ranked_sources = compact_journals(journals)
    figure('opf_leading_journals', fig, 'Publication sources appearing in the top ten by publication count, scholarly citations, mean citations or combined score. Bubble area is proportional to the value relative to the maximum within each column; printed values preserve the original units. The combined score is the equal-weight mean percentile rank of the three indicators, multiplied by 100. This explicit normalization is an adaptation of the equal-weight aggregation approach. Sources include non-journal publications because document type is unavailable for the full population.')
    ranked_sources.reset_index().to_csv(data / 'source_rankings_all.csv', index=False, encoding='utf-8-sig')
    save_table('opf_combined_source_ranking', ranked_sources.nlargest(10, 'Combined score').reset_index(),
               'Leading publication sources by combined bibliometric score.',
               'Combined score is the mean percentile rank across publication count, total scholarly citations and mean citations, multiplied by 100. Each indicator has equal weight. No minimum source publication count is imposed.')
    save_table('opf_leading_academic_sources', journals.nlargest(10, 'Publications').reset_index(),
               'Leading sources by publication volume in the DOI-linked ML--OPF population.',
               'Citations refer to the Scopus export snapshot. Source names are retained as exported.')

    # Academic citation distribution.
    bins = [-.5, .5, 9.5, 49.5, 99.5, 499.5, float('inf')]
    labels = ['0', '1--9', '10--49', '50--99', '100--499', '500+']
    hist = pd.cut(df.Scopus_citations, bins=bins, labels=labels).value_counts(sort=False)
    fig = compact_distribution(df, hist)
    figure('opf_citation_distribution', fig, 'Distribution of accumulated scholarly citations: citation bins with counts and percentages, and the share of publications attaining at least each citation count. The population is the DOI-linked export.')
    hist.rename('Publications').to_csv(data / 'academic_citation_distribution.csv', encoding='utf-8-sig')

    indicators = []
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
    fig = compact_external(df)
    figure('opf_external_citation_trends', fig, 'Patent- and policy-cited publications and their external citation totals, grouped by research publication year. These are not annual patent or policy citation-event counts; the exports have different observation dates. Bars show cited publications and lines show external citations, using identical scales across the two pathways. Shading marks records dated 2026 or later.')
    save_table('opf_citation_indicators', pd.DataFrame(indicators),
               'Patent and policy citation indicators for the DOI-linked ML--OPF population.',
               'Coverage is the percentage cited at least once; density is external citations divided by the full DOI-linked population; intensity is external citations divided by cited publications. Patent counts refer to the Lens reported patent-citation measure, not patent families. NR indicates an incomplete citation total.')
    top = df.sort_values('Scopus_citations', ascending=False).head(10)
    display = top[['Title', 'Year', 'Scopus_citations', 'patent_citation_count', 'policy_citation_count']].copy()
    display.columns = ['Publication title', 'Year', 'Scopus citations', 'Patent citations', 'Policy citations']
    save_table('opf_top_academic_cited', display, 'Ten most academically cited publications in the DOI-linked ML--OPF population.',
               'Scholarly, patent and policy citation counts come from their respective export snapshots. Zero indicates no matching citation in the supplied exports; NR denotes an unknown count.', 'Xrrrr')

    fig = compact_citations(df, data)
    figure('opf_academic_external_citations', fig, 'Academic versus external citation attention. Grey bins show the density of all publications with known counts; coloured points show externally cited publications. Circles A1--A10 identify the ten most academically cited publications and stars E1--E10 the ten most externally cited publications. Plot-label keys are exported separately for each pathway. Logarithmic transformations retain zero counts. These are citation signals rather than evidence of implementation.')

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
    fig = compact_people(authors, names, countries)
    figure('opf_authors_countries', fig, f'Productive authors identified by Scopus author ID and countries extracted from affiliation addresses. Each publication contributes once to each associated author or country. This is not first-author-country counting. Author parsing omitted {skipped} records; unparsed country segments remain excluded.')
    pd.DataFrame([{'Author ID': k, 'Author': names[k], 'Publications': v} for k, v in authors.most_common()]).to_csv(data / 'authors_full_count.csv', index=False, encoding='utf-8-sig')
    pd.Series(dict(countries), name='Publications').rename_axis('Country').to_csv(data / 'countries_full_count.csv', encoding='utf-8-sig')

    (out / 'figure_insertions.tex').write_text('\n\n'.join(fragments) + '\n', encoding='utf-8')
    lines = [r'% Requires graphicx, booktabs and tabularx. Paths assume the project root.',
             r'% Insert within your existing Results and Discussion section.',
             r'\subsection{Overview of ML--OPF Research}\label{subsec:opf_research_overview}',
             f'The DOI-linked analysis population comprised {n:,} unique publications, including {int(df.in_lens.sum()):,} patent-cited and {int(df.in_overton.sum()):,} policy-cited publications. These counts describe the exported metadata population and are distinct from the final topic-modelling population. Publications without a usable DOI are outside this bibliometric population.',
             fragments[0], fragments[1],
             r'\input{output/opf_hira_results_v2/tables/opf_leading_academic_sources.tex}',
             r'\input{output/opf_hira_results_v2/tables/opf_combined_source_ranking.tex}',
             fragments[-1], fragments[2],
             r'\input{output/opf_hira_results_v2/tables/opf_top_academic_cited.tex}',
             r'\subsection{Patent and Policy Citation Patterns}\label{subsec:opf_external_citations}',
             r'\input{output/opf_hira_results_v2/tables/opf_citation_indicators.tex}',
             fragments[3],
             r'\input{output/opf_hira_results_v2/tables/opf_patent_sources.tex}',
             r'\input{output/opf_hira_results_v2/tables/opf_policy_sources.tex}',
             r'\input{output/opf_hira_results_v2/tables/opf_top_patent_cited.tex}',
             r'\input{output/opf_hira_results_v2/tables/opf_top_policy_cited.tex}', fragments[4],
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
    archive = root / 'output/opf_hira_results_v2.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out.parent))
    print(f'\nDONE: {len(fragments)} figures (PDF + PNG), {summary["tables"]} LaTeX tables.')
    print(f'LaTeX: {out / "results_bibliometrics.tex"}\nArchive: {archive}')
    print(f'Population: {n:,}; Lens: {summary["lens"]:,}; Overton: {summary["overton"]:,}')


if __name__ == '__main__':
    main()
