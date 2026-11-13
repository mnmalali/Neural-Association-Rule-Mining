"""Classical association-rule mining baselines (Apriori, FP-Growth) and exact rule statistics."""
import time
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

from data import attr_of


def rule_key(ante, cons):
    return " & ".join(sorted(ante)) + " => " + cons


def _rules_from_itemsets(fi, n, min_conf, min_lift):
    if len(fi) == 0:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift", "count"])
    r = association_rules(fi, metric="confidence", min_threshold=min_conf, num_itemsets=n)
    r = r[(r["consequents"].map(len) == 1) & (r["lift"] >= min_lift)].copy()
    r["antecedent"] = r["antecedents"].map(lambda s: tuple(sorted(s)))
    r["consequent"] = r["consequents"].map(lambda s: next(iter(s)))
    r["count"] = (r["support"] * n).round().astype(int)
    r = r[["antecedent", "consequent", "support", "confidence", "lift", "count"]].reset_index(drop=True)
    r["key"] = [rule_key(a, c) for a, c in zip(r["antecedent"], r["consequent"])]
    return r


def mine_apriori(X, min_sup=0.05, min_conf=0.8, min_lift=1.2, max_len=4):
    t = time.time()
    fi = apriori(X, min_support=min_sup, use_colnames=True, max_len=max_len)
    r = _rules_from_itemsets(fi, len(X), min_conf, min_lift)
    r.attrs["seconds"] = time.time() - t
    return r


def mine_fpgrowth(X, min_sup=0.05, min_conf=0.8, min_lift=1.2, max_len=4):
    t = time.time()
    fi = fpgrowth(X, min_support=min_sup, use_colnames=True, max_len=max_len)
    r = _rules_from_itemsets(fi, len(X), min_conf, min_lift)
    r.attrs["seconds"] = time.time() - t
    return r


def stats_on(X, rules):
    """Recompute support / confidence / lift of given rules (antecedent tuple, consequent) on dataset X
    (used for held-out evaluation, bootstrap checks etc.)."""
    M = X.values
    col = {c: i for i, c in enumerate(X.columns)}
    n = len(X)
    out = []
    for a, c in zip(rules["antecedent"], rules["consequent"]):
        m = np.ones(n, bool)
        for it in a:
            m &= M[:, col[it]]
        na = m.sum()
        nac = (m & M[:, col[c]]).sum()
        nc = M[:, col[c]].sum()
        conf = nac / na if na else np.nan
        lift = conf / (nc / n) if nc else np.nan
        out.append((nac / n, conf, lift, nac, na))
    return pd.DataFrame(out, columns=["support", "confidence", "lift", "count", "ante_count"], index=rules.index)


def prune_unproductive(rules, X, min_gain=1e-6):
    """Remove rules that are not 'productive' (Webb): a rule A=>c is dropped if some proper subset A' of A
    (non-empty) has confidence(A'=>c) >= confidence(A=>c) - min_gain, i.e. the extra conditions add nothing."""
    if len(rules) == 0:
        return rules
    M = X.values
    col = {c: i for i, c in enumerate(X.columns)}
    n = len(X)
    cache = {}

    def conf(a, c):
        k = (a, c)
        if k not in cache:
            m = np.ones(n, bool)
            for it in a:
                m &= M[:, col[it]]
            cache[k] = (m & M[:, col[c]]).sum() / max(m.sum(), 1)
        return cache[k]

    keep = []
    from itertools import combinations
    for a, c, cf in zip(rules["antecedent"], rules["consequent"], rules["confidence"]):
        ok = True
        for k in range(1, len(a)):
            for sub in combinations(a, k):
                if conf(tuple(sorted(sub)), c) >= cf - min_gain:
                    ok = False
                    break
            if not ok:
                break
        keep.append(ok)
    return rules[np.array(keep)].reset_index(drop=True)
