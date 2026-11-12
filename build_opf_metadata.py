#!/usr/bin/env python3
"""Build DOI-linked OPF publication metadata from local Scopus/Lens/Overton exports.

Install: python -m pip install pandas openpyxl
Run from the project folder: python build_opf_metadata.py --root .
Optional: --scopus PATH --lens PATH --overton PATH --lens-dois PATH
Optional: --population PATH (DOI list or table, to restrict to modelled papers).
No network calls, title-based matching, model fitting, or modification of inputs.
Unknown values remain empty; counts describe the supplied export snapshots only.
Duplicate DOI with the same nonempty title (ignoring case/extra whitespace):
keep the first row, even if year or other metadata differs, and log removals.
Different titles sharing a DOI still require review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote

import pandas as pd

OUTPUT_COLUMNS = [
    'doi_norm', 'Title', 'Abstract', 'Year', 'Source title', 'Document Type',
    'Scopus_citations', 'Authors', 'Author_IDs', 'Affiliations',
    'Affiliation_countries', 'First_author_country', 'in_lens', 'in_overton',
    'patent_citation_count', 'policy_citation_count',
]
LENS_NAME = ('lenspatcite-export-mdzalfirdausi-52137fc2-30dc-4652-b8cb-'
             'a052ce9c5442-2026-09-27_06-13-55.xlsx')
OVERTON_NAME = 'overton_policy_doc_2026-09-10_12-24-30 - ori.csv'
DOI_RE = re.compile(r'10\.\d{4,9}/[^\s<>"\[\]{};,|]+', re.I)


def string(x):
    return '' if pd.isna(x) else str(x).strip()


def key(x):
    return re.sub(r'\s+', ' ', str(x).replace('\ufeff', '').strip()).casefold()


def doi_values(x):
    """Extract DOI tokens from DOI URLs, lists and explicit DOI-bearing identifiers.

    Semicolon/comma/pipe are interpreted as export list separators. Unusual DOI
    suffixes containing these characters require review of the unmatched audit.
    """
    value = html.unescape(unquote(string(x)))
    found = set()
    for token in DOI_RE.findall(value):
        token = token.rstrip('.').rstrip("'")
        # Preserve balanced parentheses, which occur in real DOI suffixes.
        while token.endswith(')') and token.count(')') > token.count('('):
            token = token[:-1]
        if token:
            found.add(token.casefold())
    return found


def read_table(path):
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xlsm'}:
        df = pd.read_excel(path, dtype=str, keep_default_na=False, engine='openpyxl')
    elif suffix == '.parquet':
        df = pd.read_parquet(path).fillna('').astype(str)
    else:
        df = None
        for encoding in ['utf-8-sig', 'cp1252']:
            try:
                with path.open(encoding=encoding, newline='') as handle:
                    sample = handle.read(16384)
                try:
                    delimiter = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
                except csv.Error:
                    delimiter = ','
                df = pd.read_csv(path, dtype=str, keep_default_na=False,
                                 encoding=encoding, sep=delimiter)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f'Cannot decode {path}')
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    if len({key(c) for c in df.columns}) != len(df.columns):
        raise ValueError(f'Duplicate column names in {path}')
    return df


def get_col(df, *names, required=False):
    mapping = {key(c): c for c in df.columns}
    for name in names:
        if key(name) in mapping:
            return mapping[key(name)]
    if required:
        raise ValueError(f'Missing column {names}; available: {list(df.columns)}')
    return None


def resolve(root, explicit, filename):
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ValueError(f'Input file does not exist: {path}')
        return path.resolve()
    matches = sorted(p for p in root.rglob(filename)
                     if p.is_file() and not p.name.startswith('~$'))
    if len(matches) != 1:
        raise ValueError(f'Expected one {filename} below {root}; found {matches}. '
                         'Supply its explicit command-line path.')
    return matches[0].resolve()


def nonnegative_int(x):
    value = string(x).replace(',', '')
    if not re.fullmatch(r'\d+(?:\.0+)?', value):
        return None
    return int(float(value))


# ISO 3166 country names and explicit aliases; no fuzzy address matching.
COUNTRIES = {'aruba': 'Aruba', 'afghanistan': 'Afghanistan', 'islamic republic of afghanistan': 'Afghanistan', 'angola': 'Angola', 'republic of angola': 'Angola', 'anguilla': 'Anguilla', 'åland islands': 'Åland Islands', 'albania': 'Albania', 'republic of albania': 'Albania', 'andorra': 'Andorra', 'principality of andorra': 'Andorra', 'united arab emirates': 'United Arab Emirates', 'argentina': 'Argentina', 'argentine republic': 'Argentina', 'armenia': 'Armenia', 'republic of armenia': 'Armenia', 'american samoa': 'American Samoa', 'antarctica': 'Antarctica', 'french southern territories': 'French Southern Territories', 'antigua and barbuda': 'Antigua and Barbuda', 'australia': 'Australia', 'austria': 'Austria', 'republic of austria': 'Austria', 'azerbaijan': 'Azerbaijan', 'republic of azerbaijan': 'Azerbaijan', 'burundi': 'Burundi', 'republic of burundi': 'Burundi', 'belgium': 'Belgium', 'kingdom of belgium': 'Belgium', 'benin': 'Benin', 'republic of benin': 'Benin', 'bonaire, sint eustatius and saba': 'Bonaire, Sint Eustatius and Saba', 'burkina faso': 'Burkina Faso', 'bangladesh': 'Bangladesh', "people's republic of bangladesh": 'Bangladesh', 'bulgaria': 'Bulgaria', 'republic of bulgaria': 'Bulgaria', 'bahrain': 'Bahrain', 'kingdom of bahrain': 'Bahrain', 'bahamas': 'Bahamas', 'commonwealth of the bahamas': 'Bahamas', 'bosnia and herzegovina': 'Bosnia and Herzegovina', 'republic of bosnia and herzegovina': 'Bosnia and Herzegovina', 'saint barthélemy': 'Saint Barthélemy', 'belarus': 'Belarus', 'republic of belarus': 'Belarus', 'belize': 'Belize', 'bermuda': 'Bermuda', 'bolivia, plurinational state of': 'Bolivia, Plurinational State of', 'plurinational state of bolivia': 'Bolivia, Plurinational State of', 'bolivia': 'Bolivia, Plurinational State of', 'brazil': 'Brazil', 'federative republic of brazil': 'Brazil', 'barbados': 'Barbados', 'brunei darussalam': 'Brunei Darussalam', 'bhutan': 'Bhutan', 'kingdom of bhutan': 'Bhutan', 'bouvet island': 'Bouvet Island', 'botswana': 'Botswana', 'republic of botswana': 'Botswana', 'central african republic': 'Central African Republic', 'canada': 'Canada', 'cocos (keeling) islands': 'Cocos (Keeling) Islands', 'switzerland': 'Switzerland', 'swiss confederation': 'Switzerland', 'chile': 'Chile', 'republic of chile': 'Chile', 'china': 'China', "people's republic of china": 'China', "côte d'ivoire": "Côte d'Ivoire", "republic of côte d'ivoire": "Côte d'Ivoire", 'cameroon': 'Cameroon', 'republic of cameroon': 'Cameroon', 'congo, the democratic republic of the': 'Congo, The Democratic Republic of the', 'congo': 'Congo', 'republic of the congo': 'Congo', 'cook islands': 'Cook Islands', 'colombia': 'Colombia', 'republic of colombia': 'Colombia', 'comoros': 'Comoros', 'union of the comoros': 'Comoros', 'cabo verde': 'Cabo Verde', 'republic of cabo verde': 'Cabo Verde', 'costa rica': 'Costa Rica', 'republic of costa rica': 'Costa Rica', 'cuba': 'Cuba', 'republic of cuba': 'Cuba', 'curaçao': 'Curaçao', 'christmas island': 'Christmas Island', 'cayman islands': 'Cayman Islands', 'cyprus': 'Cyprus', 'republic of cyprus': 'Cyprus', 'czechia': 'Czechia', 'czech republic': 'Czechia', 'germany': 'Germany', 'federal republic of germany': 'Germany', 'djibouti': 'Djibouti', 'republic of djibouti': 'Djibouti', 'dominica': 'Dominica', 'commonwealth of dominica': 'Dominica', 'denmark': 'Denmark', 'kingdom of denmark': 'Denmark', 'dominican republic': 'Dominican Republic', 'algeria': 'Algeria', "people's democratic republic of algeria": 'Algeria', 'ecuador': 'Ecuador', 'republic of ecuador': 'Ecuador', 'egypt': 'Egypt', 'arab republic of egypt': 'Egypt', 'eritrea': 'Eritrea', 'the state of eritrea': 'Eritrea', 'western sahara': 'Western Sahara', 'spain': 'Spain', 'kingdom of spain': 'Spain', 'estonia': 'Estonia', 'republic of estonia': 'Estonia', 'ethiopia': 'Ethiopia', 'federal democratic republic of ethiopia': 'Ethiopia', 'finland': 'Finland', 'republic of finland': 'Finland', 'fiji': 'Fiji', 'republic of fiji': 'Fiji', 'falkland islands (malvinas)': 'Falkland Islands (Malvinas)', 'france': 'France', 'french republic': 'France', 'faroe islands': 'Faroe Islands', 'micronesia, federated states of': 'Micronesia, Federated States of', 'federated states of micronesia': 'Micronesia, Federated States of', 'gabon': 'Gabon', 'gabonese republic': 'Gabon', 'united kingdom': 'United Kingdom', 'united kingdom of great britain and northern ireland': 'United Kingdom', 'georgia': 'Georgia', 'guernsey': 'Guernsey', 'ghana': 'Ghana', 'republic of ghana': 'Ghana', 'gibraltar': 'Gibraltar', 'guinea': 'Guinea', 'republic of guinea': 'Guinea', 'guadeloupe': 'Guadeloupe', 'gambia': 'Gambia', 'republic of the gambia': 'Gambia', 'guinea-bissau': 'Guinea-Bissau', 'republic of guinea-bissau': 'Guinea-Bissau', 'equatorial guinea': 'Equatorial Guinea', 'republic of equatorial guinea': 'Equatorial Guinea', 'greece': 'Greece', 'hellenic republic': 'Greece', 'grenada': 'Grenada', 'greenland': 'Greenland', 'guatemala': 'Guatemala', 'republic of guatemala': 'Guatemala', 'french guiana': 'French Guiana', 'guam': 'Guam', 'guyana': 'Guyana', 'republic of guyana': 'Guyana', 'hong kong': 'Hong Kong', 'hong kong special administrative region of china': 'Hong Kong', 'heard island and mcdonald islands': 'Heard Island and McDonald Islands', 'honduras': 'Honduras', 'republic of honduras': 'Honduras', 'croatia': 'Croatia', 'republic of croatia': 'Croatia', 'haiti': 'Haiti', 'republic of haiti': 'Haiti', 'hungary': 'Hungary', 'indonesia': 'Indonesia', 'republic of indonesia': 'Indonesia', 'isle of man': 'Isle of Man', 'india': 'India', 'republic of india': 'India', 'british indian ocean territory': 'British Indian Ocean Territory', 'ireland': 'Ireland', 'iran, islamic republic of': 'Iran, Islamic Republic of', 'islamic republic of iran': 'Iran, Islamic Republic of', 'iran': 'Iran, Islamic Republic of', 'iraq': 'Iraq', 'republic of iraq': 'Iraq', 'iceland': 'Iceland', 'republic of iceland': 'Iceland', 'israel': 'Israel', 'state of israel': 'Israel', 'italy': 'Italy', 'italian republic': 'Italy', 'jamaica': 'Jamaica', 'jersey': 'Jersey', 'jordan': 'Jordan', 'hashemite kingdom of jordan': 'Jordan', 'japan': 'Japan', 'kazakhstan': 'Kazakhstan', 'republic of kazakhstan': 'Kazakhstan', 'kenya': 'Kenya', 'republic of kenya': 'Kenya', 'kyrgyzstan': 'Kyrgyzstan', 'kyrgyz republic': 'Kyrgyzstan', 'cambodia': 'Cambodia', 'kingdom of cambodia': 'Cambodia', 'kiribati': 'Kiribati', 'republic of kiribati': 'Kiribati', 'saint kitts and nevis': 'Saint Kitts and Nevis', 'korea, republic of': 'Korea, Republic of', 'south korea': 'Korea, Republic of', 'kuwait': 'Kuwait', 'state of kuwait': 'Kuwait', "lao people's democratic republic": "Lao People's Democratic Republic", 'laos': "Lao People's Democratic Republic", 'lebanon': 'Lebanon', 'lebanese republic': 'Lebanon', 'liberia': 'Liberia', 'republic of liberia': 'Liberia', 'libya': 'Libya', 'saint lucia': 'Saint Lucia', 'liechtenstein': 'Liechtenstein', 'principality of liechtenstein': 'Liechtenstein', 'sri lanka': 'Sri Lanka', 'democratic socialist republic of sri lanka': 'Sri Lanka', 'lesotho': 'Lesotho', 'kingdom of lesotho': 'Lesotho', 'lithuania': 'Lithuania', 'republic of lithuania': 'Lithuania', 'luxembourg': 'Luxembourg', 'grand duchy of luxembourg': 'Luxembourg', 'latvia': 'Latvia', 'republic of latvia': 'Latvia', 'macao': 'Macao', 'macao special administrative region of china': 'Macao', 'saint martin (french part)': 'Saint Martin (French part)', 'morocco': 'Morocco', 'kingdom of morocco': 'Morocco', 'monaco': 'Monaco', 'principality of monaco': 'Monaco', 'moldova, republic of': 'Moldova, Republic of', 'republic of moldova': 'Moldova, Republic of', 'moldova': 'Moldova, Republic of', 'madagascar': 'Madagascar', 'republic of madagascar': 'Madagascar', 'maldives': 'Maldives', 'republic of maldives': 'Maldives', 'mexico': 'Mexico', 'united mexican states': 'Mexico', 'marshall islands': 'Marshall Islands', 'republic of the marshall islands': 'Marshall Islands', 'north macedonia': 'North Macedonia', 'republic of north macedonia': 'North Macedonia', 'mali': 'Mali', 'republic of mali': 'Mali', 'malta': 'Malta', 'republic of malta': 'Malta', 'myanmar': 'Myanmar', 'republic of myanmar': 'Myanmar', 'montenegro': 'Montenegro', 'mongolia': 'Mongolia', 'northern mariana islands': 'Northern Mariana Islands', 'commonwealth of the northern mariana islands': 'Northern Mariana Islands', 'mozambique': 'Mozambique', 'republic of mozambique': 'Mozambique', 'mauritania': 'Mauritania', 'islamic republic of mauritania': 'Mauritania', 'montserrat': 'Montserrat', 'martinique': 'Martinique', 'mauritius': 'Mauritius', 'republic of mauritius': 'Mauritius', 'malawi': 'Malawi', 'republic of malawi': 'Malawi', 'malaysia': 'Malaysia', 'mayotte': 'Mayotte', 'namibia': 'Namibia', 'republic of namibia': 'Namibia', 'new caledonia': 'New Caledonia', 'niger': 'Niger', 'republic of the niger': 'Niger', 'norfolk island': 'Norfolk Island', 'nigeria': 'Nigeria', 'federal republic of nigeria': 'Nigeria', 'nicaragua': 'Nicaragua', 'republic of nicaragua': 'Nicaragua', 'niue': 'Niue', 'netherlands': 'Netherlands', 'kingdom of the netherlands': 'Netherlands', 'norway': 'Norway', 'kingdom of norway': 'Norway', 'nepal': 'Nepal', 'federal democratic republic of nepal': 'Nepal', 'nauru': 'Nauru', 'republic of nauru': 'Nauru', 'new zealand': 'New Zealand', 'oman': 'Oman', 'sultanate of oman': 'Oman', 'pakistan': 'Pakistan', 'islamic republic of pakistan': 'Pakistan', 'panama': 'Panama', 'republic of panama': 'Panama', 'pitcairn': 'Pitcairn', 'peru': 'Peru', 'republic of peru': 'Peru', 'philippines': 'Philippines', 'republic of the philippines': 'Philippines', 'palau': 'Palau', 'republic of palau': 'Palau', 'papua new guinea': 'Papua New Guinea', 'independent state of papua new guinea': 'Papua New Guinea', 'poland': 'Poland', 'republic of poland': 'Poland', 'puerto rico': 'Puerto Rico', "korea, democratic people's republic of": "Korea, Democratic People's Republic of", "democratic people's republic of korea": "Korea, Democratic People's Republic of", 'north korea': "Korea, Democratic People's Republic of", 'portugal': 'Portugal', 'portuguese republic': 'Portugal', 'paraguay': 'Paraguay', 'republic of paraguay': 'Paraguay', 'palestine, state of': 'Palestine, State of', 'the state of palestine': 'Palestine, State of', 'french polynesia': 'French Polynesia', 'qatar': 'Qatar', 'state of qatar': 'Qatar', 'réunion': 'Réunion', 'romania': 'Romania', 'russian federation': 'Russian Federation', 'rwanda': 'Rwanda', 'rwandese republic': 'Rwanda', 'saudi arabia': 'Saudi Arabia', 'kingdom of saudi arabia': 'Saudi Arabia', 'sudan': 'Sudan', 'republic of the sudan': 'Sudan', 'senegal': 'Senegal', 'republic of senegal': 'Senegal', 'singapore': 'Singapore', 'republic of singapore': 'Singapore', 'south georgia and the south sandwich islands': 'South Georgia and the South Sandwich Islands', 'saint helena, ascension and tristan da cunha': 'Saint Helena, Ascension and Tristan da Cunha', 'svalbard and jan mayen': 'Svalbard and Jan Mayen', 'solomon islands': 'Solomon Islands', 'sierra leone': 'Sierra Leone', 'republic of sierra leone': 'Sierra Leone', 'el salvador': 'El Salvador', 'republic of el salvador': 'El Salvador', 'san marino': 'San Marino', 'republic of san marino': 'San Marino', 'somalia': 'Somalia', 'federal republic of somalia': 'Somalia', 'saint pierre and miquelon': 'Saint Pierre and Miquelon', 'serbia': 'Serbia', 'republic of serbia': 'Serbia', 'south sudan': 'South Sudan', 'republic of south sudan': 'South Sudan', 'sao tome and principe': 'Sao Tome and Principe', 'democratic republic of sao tome and principe': 'Sao Tome and Principe', 'suriname': 'Suriname', 'republic of suriname': 'Suriname', 'slovakia': 'Slovakia', 'slovak republic': 'Slovakia', 'slovenia': 'Slovenia', 'republic of slovenia': 'Slovenia', 'sweden': 'Sweden', 'kingdom of sweden': 'Sweden', 'eswatini': 'Eswatini', 'kingdom of eswatini': 'Eswatini', 'sint maarten (dutch part)': 'Sint Maarten (Dutch part)', 'seychelles': 'Seychelles', 'republic of seychelles': 'Seychelles', 'syrian arab republic': 'Syrian Arab Republic', 'syria': 'Syrian Arab Republic', 'turks and caicos islands': 'Turks and Caicos Islands', 'chad': 'Chad', 'republic of chad': 'Chad', 'togo': 'Togo', 'togolese republic': 'Togo', 'thailand': 'Thailand', 'kingdom of thailand': 'Thailand', 'tajikistan': 'Tajikistan', 'republic of tajikistan': 'Tajikistan', 'tokelau': 'Tokelau', 'turkmenistan': 'Turkmenistan', 'timor-leste': 'Timor-Leste', 'democratic republic of timor-leste': 'Timor-Leste', 'tonga': 'Tonga', 'kingdom of tonga': 'Tonga', 'trinidad and tobago': 'Trinidad and Tobago', 'republic of trinidad and tobago': 'Trinidad and Tobago', 'tunisia': 'Tunisia', 'republic of tunisia': 'Tunisia', 'türkiye': 'Türkiye', 'republic of türkiye': 'Türkiye', 'tuvalu': 'Tuvalu', 'taiwan, province of china': 'Taiwan, Province of China', 'taiwan': 'Taiwan, Province of China', 'tanzania, united republic of': 'Tanzania, United Republic of', 'united republic of tanzania': 'Tanzania, United Republic of', 'tanzania': 'Tanzania, United Republic of', 'uganda': 'Uganda', 'republic of uganda': 'Uganda', 'ukraine': 'Ukraine', 'united states minor outlying islands': 'United States Minor Outlying Islands', 'uruguay': 'Uruguay', 'eastern republic of uruguay': 'Uruguay', 'united states': 'United States', 'united states of america': 'United States', 'uzbekistan': 'Uzbekistan', 'republic of uzbekistan': 'Uzbekistan', 'holy see (vatican city state)': 'Holy See (Vatican City State)', 'saint vincent and the grenadines': 'Saint Vincent and the Grenadines', 'venezuela, bolivarian republic of': 'Venezuela, Bolivarian Republic of', 'bolivarian republic of venezuela': 'Venezuela, Bolivarian Republic of', 'venezuela': 'Venezuela, Bolivarian Republic of', 'virgin islands, british': 'Virgin Islands, British', 'british virgin islands': 'Virgin Islands, British', 'virgin islands, u.s.': 'Virgin Islands, U.S.', 'virgin islands of the united states': 'Virgin Islands, U.S.', 'viet nam': 'Viet Nam', 'socialist republic of viet nam': 'Viet Nam', 'vietnam': 'Viet Nam', 'vanuatu': 'Vanuatu', 'republic of vanuatu': 'Vanuatu', 'wallis and futuna': 'Wallis and Futuna', 'samoa': 'Samoa', 'independent state of samoa': 'Samoa', 'yemen': 'Yemen', 'republic of yemen': 'Yemen', 'south africa': 'South Africa', 'republic of south africa': 'South Africa', 'zambia': 'Zambia', 'republic of zambia': 'Zambia', 'zimbabwe': 'Zimbabwe', 'republic of zimbabwe': 'Zimbabwe'}
COUNTRIES.update({
    'usa': 'United States', 'u.s.a.': 'United States', 'us': 'United States',
    'uk': 'United Kingdom', 'u.k.': 'United Kingdom', 'england': 'United Kingdom',
    'scotland': 'United Kingdom', 'wales': 'United Kingdom',
    'northern ireland': 'United Kingdom', 'south korea': 'Korea, Republic of',
    'republic of korea': 'Korea, Republic of', 'iran': 'Iran, Islamic Republic of',
    'russia': 'Russian Federation', 'vietnam': 'Viet Nam',
    'taiwan': 'Taiwan, Province of China', 'turkey': 'Türkiye',
    'peoples r china': 'China', "people's republic of china": 'China',
    'hong kong': 'Hong Kong', 'macao': 'Macao',
})


def affiliation_countries(value):
    """Conservative exact country match at the end of each affiliation segment."""
    matched, unmatched = set(), []
    for segment in re.split(r';|\n', string(value)):
        if not segment.strip():
            continue
        tail = segment.rsplit(',', 1)[-1].strip().rstrip('.')
        country = COUNTRIES.get(key(tail))
        if country:
            matched.add(country)
        else:
            unmatched.append(segment.strip())
    return '; '.join(sorted(matched)), unmatched


def write_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', na_rep='')


def run(args):
    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve() if args.out else root / 'output' / 'opf_metadata'
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        'scopus': resolve(root, args.scopus, 'scopus_full.xlsx'),
        'lens': resolve(root, args.lens, LENS_NAME),
        'overton': resolve(root, args.overton, OVERTON_NAME),
        'lens_dois': resolve(root, args.lens_dois, 'lens_patent_cited_dois.txt'),
    }
    for name, path in paths.items():
        print(f'{name}: {path}')
    scopus = read_table(paths['scopus'])
    lens = read_table(paths['lens'])
    overton = read_table(paths['overton'])
    audit = {'input_files': {k: str(v) for k, v in paths.items()},
             'input_rows': {'scopus': len(scopus), 'lens': len(lens), 'overton': len(overton)},
             'warnings': []}

    def warn(message):
        audit['warnings'].append(message)
        print('WARNING:', message)

    doi_col = get_col(scopus, 'DOI', 'doi_norm', required=True)
    candidates = scopus[doi_col].map(doi_values)
    invalid = candidates.map(len).ne(1)
    write_csv(scopus[invalid], outdir / 'scopus_missing_or_ambiguous_doi.csv')
    audit['scopus_rows_without_single_doi'] = int(invalid.sum())
    scopus = scopus.loc[~invalid].copy()
    scopus['doi_norm'] = candidates.loc[~invalid].map(lambda x: next(iter(x)))
    original_rows = len(scopus)
    scopus = scopus.drop_duplicates()
    audit['scopus_identical_rows_removed'] = original_rows - len(scopus)
    duplicates = scopus['doi_norm'].duplicated(keep=False)
    write_csv(scopus[duplicates], outdir / 'scopus_duplicate_dois.csv')
    title_col = get_col(scopus, 'Title', required=True)
    conflicting_dois = set()
    for doi, group in scopus[duplicates].groupby('doi_norm', sort=False):
        titles = group[title_col].map(key)
        if titles.eq('').any() or titles.nunique() != 1:
            conflicting_dois.add(doi)
    conflicts = scopus[scopus['doi_norm'].isin(conflicting_dois)]
    write_csv(conflicts, outdir / 'scopus_conflicting_titles.csv')
    removed = scopus[scopus['doi_norm'].duplicated(keep='first') &
                     ~scopus['doi_norm'].isin(conflicting_dois)]
    write_csv(removed, outdir / 'scopus_same_title_duplicates_removed.csv')
    audit['scopus_same_title_rows_removed'] = len(removed)
    audit['scopus_duplicate_policy'] = (
        'Keep first row for the same normalized DOI and same nonempty title; '
        'title comparison ignores case and repeated whitespace. Other metadata '
        'differences do not block this rule. Different/missing titles are blocked.')
    if conflicting_dois:
        raise ValueError('Duplicate Scopus DOIs with different/missing titles: inspect scopus_conflicting_titles.csv. '
                         'No publication metadata output was written by this run.')
    scopus = scopus.drop_duplicates('doi_norm', keep='first').copy()
    if len(removed):
        print(f'Deduplication: kept first row for matching DOI/title; removed {len(removed)} duplicate rows.')

    output = pd.DataFrame({'doi_norm': scopus['doi_norm']}, index=scopus.index)
    fields = {
        'Title': ('Title',), 'Abstract': ('Abstract',), 'Year': ('Year',),
        'Source title': ('Source title',), 'Document Type': ('Document Type',),
        'Scopus_citations': ('Cited by', 'Scopus_citations'),
        'Authors': ('Authors',), 'Author_IDs': ('Author(s) ID', 'Author_IDs'),
        'Affiliations': ('Affiliations',),
        'First_author_country': ('First_author_country', 'First author country'),
    }
    for destination, names in fields.items():
        column = get_col(scopus, *names)
        output[destination] = scopus[column].map(string) if column else ''
        if column is None:
            warn(f'{destination}: no source column; left blank.')
    for name in ['Year', 'Scopus_citations']:
        original = output[name].copy()
        output[name] = pd.array(original.map(nonnegative_int), dtype='Int64')
        bad = original.ne('') & output[name].isna()
        write_csv(pd.DataFrame({'doi_norm': output.loc[bad, 'doi_norm'],
                                'field': name, 'raw_value': original[bad]}),
                  outdir / f'invalid_{name}.csv')
        if bad.any():
            warn(f'{name}: {int(bad.sum())} invalid values left blank; inspect audit CSV.')

    country_rows, unknown_countries = [], []
    for row in output.itertuples(index=False):
        countries, missing = affiliation_countries(row.Affiliations)
        country_rows.append(countries)
        for affiliation in missing:
            unknown_countries.append({'doi_norm': row.doi_norm, 'unparsed_affiliation': affiliation})
    output['Affiliation_countries'] = country_rows
    write_csv(pd.DataFrame(unknown_countries, columns=['doi_norm', 'unparsed_affiliation']),
              outdir / 'unparsed_affiliations.csv')
    if unknown_countries:
        warn(f'{len(unknown_countries)} affiliation segments have no recognized country suffix; '
             'country lists may be incomplete. Review unparsed_affiliations.csv.')

    lens_list = doi_values(paths['lens_dois'].read_text(encoding='utf-8-sig'))
    lens_members = set(lens_list)
    lens_reported = defaultdict(set)
    lens_unknown, lens_bad = set(), []
    external = get_col(lens, 'citation external id', required=True)
    citation = get_col(lens, 'citation string')
    count_col = get_col(lens, 'cited by patent count', required=True)
    for i, row in lens.iterrows():
        ds = doi_values(row[external])
        if not ds and citation:
            ds = doi_values(row[citation])
        if len(ds) != 1:
            lens_bad.append({'source_row': int(i) + 2, 'reason': 'No single DOI', **row.to_dict()})
            continue
        doi = next(iter(ds))
        lens_members.add(doi)
        count = nonnegative_int(row[count_col])
        if count is None or count == 0:
            lens_unknown.add(doi)
            lens_bad.append({'source_row': int(i) + 2,
                             'reason': 'Missing/invalid/zero count in patent-cited export', **row.to_dict()})
        else:
            lens_reported[doi].add(count)
    write_csv(pd.DataFrame(lens_bad, columns=['source_row', 'reason'] + list(lens.columns)),
              outdir / 'lens_rows_for_review.csv')
    conflicts = [{'doi_norm': d, 'reported_counts': '; '.join(map(str, sorted(v)))}
                 for d, v in lens_reported.items() if len(v) > 1]
    write_csv(pd.DataFrame(conflicts, columns=['doi_norm', 'reported_counts']),
              outdir / 'lens_count_conflicts.csv')
    # Repeated reported citation totals must never be summed.
    lens_counts = {d: next(iter(v)) for d, v in lens_reported.items()
                   if len(v) == 1 and d not in lens_unknown}

    cited_col = get_col(overton, 'Cited DOIs', required=True)
    policy_id = get_col(overton, 'Overton document ID', required=True)
    policy_members, policy_missing_id = set(), set()
    policy_ids = defaultdict(set)
    policy_bad, edges = [], []
    for i, row in overton.iterrows():
        ds = doi_values(row[cited_col])
        pid = string(row[policy_id])
        if not ds:
            policy_bad.append({'source_row': int(i) + 2, 'reason': 'No DOI parsed from Cited DOIs',
                               **row.to_dict()})
        policy_members.update(ds)
        for doi in sorted(ds):
            if pid:
                policy_ids[doi].add(pid)
                edges.append({'doi_norm': doi, 'Overton document ID': pid,
                              'Document title': row.get('Document title', ''),
                              'Publication date': row.get('Publication date', ''),
                              'Source name': row.get('Source name', ''),
                              'Source country': row.get('Source country', '')})
            else:
                policy_missing_id.add(doi)
        if ds and not pid:
            policy_bad.append({'source_row': int(i) + 2, 'reason': 'Missing Overton document ID',
                               **row.to_dict()})
    write_csv(pd.DataFrame(policy_bad, columns=['source_row', 'reason'] + list(overton.columns)),
              outdir / 'overton_rows_for_review.csv')
    edge_cols = ['doi_norm', 'Overton document ID', 'Document title', 'Publication date',
                 'Source name', 'Source country']
    write_csv(pd.DataFrame(edges, columns=edge_cols).drop_duplicates(['doi_norm', 'Overton document ID']),
              outdir / 'policy_citation_edges.csv')

    output['in_lens'] = output['doi_norm'].isin(lens_members)
    output['in_overton'] = output['doi_norm'].isin(policy_members)
    output['patent_citation_count'] = pd.array(
        [lens_counts.get(d) if d in lens_members else 0 for d in output['doi_norm']], dtype='Int64')
    output['policy_citation_count'] = pd.array(
        [None if d in policy_missing_id else len(policy_ids.get(d, set()))
         for d in output['doi_norm']], dtype='Int64')

    # Membership is relative to these files, not a claim of universal citation absence.
    if args.population:
        path = Path(args.population).expanduser().resolve()
        if path.suffix.lower() == '.txt':
            wanted = sorted(doi_values(path.read_text(encoding='utf-8-sig')))
        else:
            population = read_table(path)
            pcol = get_col(population, 'doi_norm', 'DOI', required=True)
            parsed = population[pcol].map(doi_values)
            if parsed.map(len).ne(1).any():
                raise ValueError('Population contains missing or ambiguous DOIs.')
            wanted = parsed.map(lambda ds: next(iter(ds))).tolist()
        if len(wanted) != len(set(wanted)):
            raise ValueError('Population contains duplicate DOIs.')
        absent = sorted(set(wanted) - set(output['doi_norm']))
        write_csv(pd.DataFrame({'doi_norm': absent}), outdir / 'population_missing_from_scopus.csv')
        if absent:
            raise ValueError('Some population DOIs are absent from Scopus. See population_missing_from_scopus.csv.')
        output = output.set_index('doi_norm').loc[wanted].reset_index()
        audit['population_file'] = str(path)
    else:
        warn('No --population provided: output uses all unique valid Scopus DOIs, '
             'not automatically the 15,743 modelled publications.')

    scopus_set = set(scopus['doi_norm'])
    for source, members in [('lens', lens_members), ('overton', policy_members)]:
        write_csv(pd.DataFrame({'doi_norm': sorted(members - scopus_set)}),
                  outdir / f'{source}_dois_not_in_scopus.csv')
    missing_counts = output[output['in_lens'] & output['patent_citation_count'].isna()][['doi_norm', 'Title']]
    write_csv(missing_counts, outdir / 'lens_members_without_verified_count.csv')
    if len(missing_counts):
        warn(f'{len(missing_counts)} patent-cited publications have no verified count; counts left blank.')
    if lens_bad:
        warn(f'{len(lens_bad)} Lens rows need review; see lens_rows_for_review.csv.')
    if policy_bad:
        warn(f'{len(policy_bad)} Overton rows need review; memberships/counts may be incomplete.')
    if invalid.any():
        warn(f'{int(invalid.sum())} Scopus rows excluded for missing/ambiguous DOI; see audit CSV.')
    output = output[OUTPUT_COLUMNS]
    target = outdir / 'opf_publication_metadata.csv'
    write_csv(output, target)
    audit.update({
        'output_rows': len(output), 'output_columns': OUTPUT_COLUMNS,
        'lens_doi_list_unique': len(lens_list),
        'lens_matched_output': int(output['in_lens'].sum()),
        'overton_matched_output': int(output['in_overton'].sum()),
        'both_matched_output': int((output['in_lens'] & output['in_overton']).sum()),
        'missing_values': {c: int((output[c].isna() | output[c].astype(str).str.strip().eq('')).sum())
                           for c in OUTPUT_COLUMNS},
        'definitions': {
            'in_lens': 'DOI in patent-cited DOI list or matched Lens export row',
            'in_overton': 'DOI explicitly in Cited DOIs of a supplied policy record',
            'patent_citation_count': 'Consistent Lens cited by patent count; repeated totals never summed',
            'policy_citation_count': 'Distinct Overton document IDs citing the DOI in this export',
            'zero_counts': 'No matched citation in supplied exports; not verified global zero',
            'First_author_country': 'Explicit source field only; never guessed from affiliation order',
            'Affiliation_countries': 'Exact country suffixes parsed from affiliation segments; review audit',
        },
    })
    (outdir / 'metadata_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nREADY: {target}\nPublications: {len(output):,}')
    print(f'Lens matched: {audit["lens_matched_output"]:,}; '
          f'Overton matched: {audit["overton_matched_output"]:,}')
    print(f'Audit: {outdir / "metadata_audit.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='.', help='Project folder; default: current folder')
    for option in ['scopus', 'lens', 'overton', 'lens-dois']:
        parser.add_argument('--' + option, help='Explicit input file path; otherwise discovered below --root')
    parser.add_argument('--population', help='Optional modelled DOI population: TXT/CSV/XLSX/Parquet')
    parser.add_argument('--out', help='Output folder; default: ROOT/output/opf_metadata')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
