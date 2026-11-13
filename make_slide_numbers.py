"""Exports results/slide_numbers.json used by build_slides.js (run after run_all.py)."""
import json, pandas as pd
S = json.load(open('results/summary.json')); R = lambda f: pd.read_csv(f'results/{f}.csv')
e1 = R('e1_main_comparison').set_index('method'); e2 = R('e2_heldout_accuracy').set_index('method'); e4 = R('e4_stability')
ds = R('e7_downstream_fairness'); pr = R('e8_privacy_utility'); fdr = R('e5_fdr_productivity'); und = R('e6_understandability')
e3 = R('e3_sample_size_summary'); cards = R('e6_rule_cards')
N = {'n_rows': S['n_rows'], 'n_items': S['n_items'], 'n_train': S['n_train'], 'n_test': S['n_test'], 'ntxn': S['avg_items_per_txn'], 'pos': S['pos_rate'],
     'e1': {m: {k: (None if pd.isna(v) else float(v)) for k, v in r.items()} for m, r in e1.iterrows()},
     'e2': {m: {k: float(v) for k, v in r.items()} for m, r in e2.iterrows()},
     'stab': [{k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.items()} for r in e4.to_dict('records')],
     'ds': ds.where(ds.notna(), None).to_dict('records'), 'pr': pr.to_dict('records'), 'fdr': fdr.to_dict('records'), 'und': und.to_dict('records'),
     'e3': e3.where(e3.notna(), None).to_dict('records'), 'null': S['e5_null'], 'audit': S['audit'], 'gap': S['base_rate_gap'], 'kanon': S['k_anonymity'],
     'cards': cards[['rule', 'confidence', 'lift', 'explanation']].to_dict('records'), 'corr': S['ae_corr_p_conf'], 'mae': S['ae_mae_p_conf'],
     'fail': S['proposal_failures'], 'train_s': S['ae_train_seconds']}
json.dump(N, open('results/slide_numbers.json', 'w'), default=float)
