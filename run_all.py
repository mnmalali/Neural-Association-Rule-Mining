"""Reproduces every number and figure of the project.   Usage:  python run_all.py   (~10-15 min on 2 CPU cores)
All randomness is seeded (SEED=42); results are written to results/*.csv and results/summary.json, figures to figures/."""
import json, os, sys, time, warnings
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from statsmodels.stats.multitest import multipletests
from mlxtend.frequent_patterns import apriori

import data as D
import mining as MN
import neural as NN
import explain as EX

SEED = 42
MIN_SUP, MIN_CONF, MIN_LIFT = 0.05, 0.80, 1.20     # rule thresholds (all methods)
TAU, SLACK = 0.75, 0.05                             # neural proposal threshold / slack
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
S = {}                                              # summary dictionary written to results/summary.json
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_NN, C_AP, C_RND, C_CONS = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def save(name):
    plt.tight_layout()
    path = os.path.abspath(os.path.join("figures", f"{name}.png"))
    # OneDrive and other sync clients can briefly lock an existing PNG while
    # it is being replaced. Retry the write so a reproducible run is robust
    # when the repository lives in a synced Windows folder.
    for attempt in range(5):
        try:
            plt.savefig(path, bbox_inches="tight")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1)
    plt.close()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def n_epochs(n):
    batch = 64 if n < 5000 else 256
    return batch, int(np.clip(3000 / np.ceil(n / batch), 40, 300))


def neural_pipeline(Xs, seed, tau=TAU, slack=SLACK, untrained=False, h2=16, min_sup=MIN_SUP, val=None):
    """Train a DAE on Xs and extract rules. Returns (verified rules, all proposals, ae)."""
    batch, ep = n_epochs(len(Xs))
    ae = NN.DenoisingAE(Xs.shape[1], 64, h2, seed)
    if not untrained:
        ae.fit(Xs.values, epochs=ep, batch=batch, val=None if val is None else val.values)
    rules, cand = NN.neural_rules(ae, Xs, min_sup=min_sup, min_conf=MIN_CONF, min_lift=MIN_LIFT, tau=tau, slack=slack)
    return rules, cand, ae


def consensus(rules):
    """Neural-consensus rules: verified by the data AND the network itself is confident (p_nn >= min_conf)."""
    return rules[rules["p_nn"] >= MIN_CONF].reset_index(drop=True) if len(rules) else rules


t_start = time.time()
raw, cat, X = D.load_transactions("data/adult-all.csv")
y = X["income=>50K"].values
idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=SEED, stratify=y)
Xtr, Xte = X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True)
cat_tr = cat.iloc[idx_tr].reset_index(drop=True)
S["n_rows"], S["n_items"], S["n_train"], S["n_test"] = len(X), X.shape[1], len(Xtr), len(Xte)
S["avg_items_per_txn"] = float(X.sum(axis=1).mean())
S["pos_rate"] = float(y.mean())
S["missing"] = {k: int(v) for k, v in raw.isna().sum().items() if v > 0}

# ---------------------------------------------------------------- dataset description table
desc = []
for a in cat.columns:
    vc = cat[a].value_counts()
    desc.append({"attribute": a, "n_values": len(vc), "values (top 4)": ", ".join(map(str, vc.index[:4])),
                 "missing": int(cat[a].isna().sum()), "sensitive": a in D.SENSITIVE_ATTRS})
pd.DataFrame(desc).to_csv("results/dataset_description.csv", index=False)

# ---------------------------------------------------------------- E1: neural vs classical on the full training set
print("E1 main comparison", flush=True)
ap = MN.mine_apriori(Xtr, MIN_SUP, MIN_CONF, MIN_LIFT)
fp = MN.mine_fpgrowth(Xtr, MIN_SUP, MIN_CONF, MIN_LIFT)
t0 = time.time()
nn_rules, nn_cand, ae = neural_pipeline(Xtr, seed=SEED, val=Xte)
nn_secs = time.time() - t0
rnd_rules, rnd_cand, _ = neural_pipeline(Xtr, seed=SEED, untrained=True)
K_ap = set(ap["key"])
cons_rules = consensus(nn_rules)
rows = []
for name, r, cand in [("Apriori", ap, None), ("FP-Growth", fp, None), ("Neural (verified)", nn_rules, nn_cand),
                      ("Neural consensus", cons_rules, nn_cand), ("Untrained AE (ablation)", rnd_rules, rnd_cand)]:
    k = set(r["key"])
    rows.append({"method": name, "n_rules": len(r), "recall_vs_apriori": len(k & K_ap) / len(K_ap),
                 "precision_vs_apriori": len(k & K_ap) / max(len(k), 1),
                 "proposals": 0 if cand is None else len(cand),
                 "proposal_precision": np.nan if cand is None else float(cand["verified"].mean()),
                 "mining_seconds": r.attrs.get("seconds", np.nan), "mean_conf": r["confidence"].mean(),
                 "mean_lift": r["lift"].mean(), "mean_support": r["support"].mean()})
main = pd.DataFrame(rows)
main.to_csv("results/e1_main_comparison.csv", index=False)
S["e1"] = main.set_index("method").to_dict("index")
S["ae_train_seconds"] = nn_secs
S["ap_fp_identical"] = bool(set(ap["key"]) == set(fp["key"]))
hist = pd.DataFrame(ae.history)
hist.to_csv("results/ae_training_history.csv", index=False)
plt.figure(figsize=(4.6, 3))
plt.plot(hist["epoch"], hist["train_loss"], label="train (masked input)", color=C_NN)
plt.plot(hist["epoch"], hist["val_loss"], label="held-out", color=C_AP, ls="--")
plt.xlabel("epoch"); plt.ylabel("reconstruction BCE"); plt.legend(frameon=False); plt.title("Autoencoder training")
save("fig_training_loss")

# proposal-error analysis: why does the network propose rules that fail verification?
bad = nn_cand[~nn_cand["verified"]]
S["proposal_failures"] = {"fail_support": int((bad["support"] < MIN_SUP).sum()), "fail_confidence": int((bad["confidence"] < MIN_CONF).sum()),
                          "fail_lift": int((bad["lift"] < MIN_LIFT).sum()), "n_bad": int(len(bad))}
# AE calibration: predicted P(c|A) vs exact confidence of the same (A,c)
bins = np.linspace(0.4, 1.0, 7)
cal = nn_cand.copy()
cal["bin"] = pd.cut(cal["p_nn"], bins, include_lowest=True)
calt = cal.groupby("bin", observed=True).agg(pred=("p_nn", "mean"), true=("confidence", "mean"), n=("p_nn", "size")).reset_index()
calt.to_csv("results/ae_calibration.csv", index=False)
S["ae_corr_p_conf"] = float(np.corrcoef(nn_cand["p_nn"], nn_cand["confidence"])[0, 1])
S["ae_mae_p_conf"] = float((nn_cand["p_nn"] - nn_cand["confidence"]).abs().mean())
plt.figure(figsize=(4.4, 3.6))
plt.scatter(nn_cand["p_nn"], nn_cand["confidence"], s=5, alpha=.25, color=C_NN)
plt.plot([.4, 1], [.4, 1], "k--", lw=1)
plt.plot(calt["pred"], calt["true"], "o-", color=C_AP, label="binned mean")
plt.xlabel("network estimate  P(c | A)"); plt.ylabel("exact confidence on data"); plt.legend(frameon=False)
plt.title("Autoencoder calibration (error analysis)")
save("fig_calibration")

# sensitivity: tau / slack grid and bottleneck size (ablations)
ante_tr = NN.frequent_antecedents(Xtr, MIN_SUP, 3)
grid = []
for tau in (0.6, 0.7, 0.75, 0.8, 0.9):
    for sl in (0.0, 0.05, 0.1):
        r, c = NN.neural_rules(ae, Xtr, MIN_SUP, MIN_CONF, MIN_LIFT, tau=tau, slack=sl, ante=ante_tr)
        grid.append({"tau": tau, "slack": sl, "proposals": len(c), "precision": float(c["verified"].mean()) if len(c) else np.nan,
                     "recall": len(set(r["key"]) & K_ap) / len(K_ap)})
pd.DataFrame(grid).to_csv("results/e1_tau_slack_grid.csv", index=False)
bn = []
for h2 in (4, 8, 16, 32):
    r, c, _ = neural_pipeline(Xtr, SEED, h2=h2)
    bn.append({"bottleneck": h2, "proposals": len(c), "precision": float(c["verified"].mean()), "recall": len(set(r["key"]) & K_ap) / len(K_ap)})
pd.DataFrame(bn).to_csv("results/e1_bottleneck.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
g = pd.DataFrame(grid)
for sl, m in zip((0.0, 0.05, 0.1), "o^s"):
    gg = g[g.slack == sl]
    ax[0].plot(gg["recall"], gg["precision"], m + "-", label=f"slack={sl}")
    for _, rr in gg.iterrows():
        ax[0].annotate(f"{rr.tau}", (rr.recall, rr.precision), fontsize=6, xytext=(2, 2), textcoords="offset points")
ax[0].set_xlabel("recall of exact Apriori rules"); ax[0].set_ylabel("proposal precision"); ax[0].legend(frameon=False)
ax[0].set_title("Threshold sweep (labels = tau)")
b = pd.DataFrame(bn)
ax[1].plot(b["bottleneck"], b["recall"], "o-", label="recall", color=C_NN)
ax[1].plot(b["bottleneck"], b["precision"], "s--", label="precision", color=C_AP)
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("bottleneck size"); ax[1].legend(frameon=False); ax[1].set_title("Bottleneck ablation")
save("fig_sensitivity")

# ---------------------------------------------------------------- E2: held-out accuracy of rules
print("E2 held-out accuracy", flush=True)
acc_rows = []
raw_nn = nn_cand.copy()                                   # raw (unverified) network proposals
raw_rnd = rnd_cand.copy()
for name, r in [("Apriori", ap), ("Neural (verified)", nn_rules), ("Neural consensus", cons_rules)]:
    st = MN.stats_on(Xte, r)
    acc_rows.append({"method": name, "n_rules": len(r), "train_conf": r["confidence"].mean(), "test_conf": st["confidence"].mean(),
                     "conf_MAE_train_vs_test": float((st["confidence"] - r["confidence"]).abs().mean()),
                     "test_support_MAE": float((st["support"] - r["support"]).abs().mean()),
                     "replication_rate": float(((st["confidence"] >= MIN_CONF - 0.05) & (st["lift"] >= 1.1)).mean())})
for name, c in [("Neural raw proposals (no verification)", raw_nn), ("Untrained AE raw (ablation)", raw_rnd)]:
    st = MN.stats_on(Xte, c)
    # conf. estimated by the network (p_nn) vs. test confidence
    acc_rows.append({"method": name, "n_rules": len(c), "train_conf": c["p_nn"].mean(), "test_conf": st["confidence"].mean(),
                     "conf_MAE_train_vs_test": float((st["confidence"] - c["p_nn"]).abs().mean()),
                     "test_support_MAE": float((st["support"] - c["support"]).abs().mean()),
                     "replication_rate": float(((st["confidence"] >= MIN_CONF - 0.05) & (st["lift"] >= 1.1) & (st["support"] >= MIN_SUP * 0.8)).mean())})
acc = pd.DataFrame(acc_rows)
acc.to_csv("results/e2_heldout_accuracy.csv", index=False)
S["e2"] = acc.set_index("method").to_dict("index")
# which rules degrade most on test (error analysis)
st_ap = MN.stats_on(Xte, ap)
err = ap.copy()
err["test_conf"], err["test_lift"], err["drop"] = st_ap["confidence"], st_ap["lift"], ap["confidence"] - st_ap["confidence"]
err["ante_len"] = err["antecedent"].map(len)
err["cons_attr"] = err["consequent"].map(D.attr_of)
err.sort_values("drop", ascending=False).head(10)[["key", "confidence", "test_conf", "drop", "count"]].to_csv("results/e2_worst_rules.csv", index=False)
by_len = err.groupby("ante_len").agg(n=("key", "size"), mean_drop=("drop", "mean"), max_drop=("drop", "max")).reset_index()
by_len.to_csv("results/e2_drop_by_antecedent_length.csv", index=False)
by_c = err.groupby("cons_attr").agg(n=("key", "size"), mean_drop=("drop", "mean")).reset_index().sort_values("n", ascending=False)
by_c.to_csv("results/e2_drop_by_consequent_attribute.csv", index=False)
S["e2_drop_by_len"] = by_len.set_index("ante_len").to_dict("index")

# ---------------------------------------------------------------- E3: sample-size sweep (spurious rules at small n)
print("E3 sample-size sweep", flush=True)
sweep = []
for n in (500, 1000, 2000, 5000, 10000, len(Xtr)):
    for rep in range(5 if n < len(Xtr) else 1):
        rs = np.random.default_rng(SEED + rep)
        sub = Xtr.iloc[rs.choice(len(Xtr), n, replace=False)].reset_index(drop=True) if n < len(Xtr) else Xtr
        a = MN.mine_apriori(sub, MIN_SUP, MIN_CONF, MIN_LIFT)
        r, c, _ = neural_pipeline(sub, SEED + rep)
        cr = consensus(r)
        for name, rr in [("Apriori", a), ("Neural (verified)", r), ("Neural consensus", cr)]:
            if len(rr) == 0:
                sweep.append({"n": n, "rep": rep, "method": name, "n_rules": 0, "replication": np.nan, "conf_err": np.nan})
                continue
            st = MN.stats_on(Xte, rr)
            sweep.append({"n": n, "rep": rep, "method": name, "n_rules": len(rr),
                          "replication": float(((st["confidence"] >= MIN_CONF - 0.05) & (st["lift"] >= 1.1)).mean()),
                          "conf_err": float((st["confidence"] - rr["confidence"]).abs().mean())})
sw = pd.DataFrame(sweep)
sw.to_csv("results/e3_sample_size_sweep.csv", index=False)
swm = sw.groupby(["method", "n"]).agg(n_rules=("n_rules", "mean"), replication=("replication", "mean"), repl_sd=("replication", "std"),
                                      conf_err=("conf_err", "mean")).reset_index()
swm.to_csv("results/e3_sample_size_summary.csv", index=False)
S["e3"] = {m: g.set_index("n")[["n_rules", "replication", "conf_err"]].to_dict("index") for m, g in swm.groupby("method")}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for m, col in [("Apriori", C_AP), ("Neural (verified)", C_NN), ("Neural consensus", C_CONS)]:
    g = swm[swm.method == m]
    ax[0].plot(g["n"], g["n_rules"], "o-", color=col, label=m)
    ax[1].errorbar(g["n"], g["replication"], yerr=g["repl_sd"].fillna(0), fmt="o-", color=col, label=m, capsize=2)
    ax[2].plot(g["n"], g["conf_err"], "o-", color=col, label=m)
for a_, t in zip(ax, ["# rules mined", "held-out replication rate", "|conf(train) - conf(test)|"]):
    a_.set_xscale("log"); a_.set_xlabel("training transactions"); a_.set_title(t)
ax[1].legend(frameon=False, fontsize=8)
save("fig_sample_size")

# ---------------------------------------------------------------- E4: stability
print("E4 stability", flush=True)
stab_rows, retention = [], {}
B = 8
for label, n in [("full train", len(Xtr)), ("n=2000", 2000)]:
    sets = {"Apriori": [], "Neural (verified)": [], "Neural consensus": []}
    lifts = {"Apriori": [], "Neural (verified)": []}
    for b in range(B):
        rs = np.random.default_rng(1000 + b)
        sub = Xtr.iloc[rs.choice(len(Xtr), n, replace=True)].reset_index(drop=True)   # bootstrap resample
        a = MN.mine_apriori(sub, MIN_SUP, MIN_CONF, MIN_LIFT)
        r, c, _ = neural_pipeline(sub, 2000 + b)                                       # different seed too
        sets["Apriori"].append(a); sets["Neural (verified)"].append(r); sets["Neural consensus"].append(consensus(r))
    for m, lst in sets.items():
        js = [jaccard(x["key"], y_["key"]) for x, y_ in combinations(lst, 2)]
        stab_rows.append({"setting": label, "method": m, "mean_jaccard": float(np.mean(js)), "sd_jaccard": float(np.std(js)),
                          "mean_n_rules": float(np.mean([len(x) for x in lst]))})
        cnt = pd.Series([k for x in lst for k in x["key"]]).value_counts() / B
        retention[(label, m)] = float((cnt >= 0.9).sum() / max(len(cnt), 1))      # share of ever-seen rules found in >=90% of runs
        stab_rows[-1]["share_rules_in_90pct_runs"] = retention[(label, m)]
# seed-only stability of the network (same data, different initialisation)
seed_sets = [neural_pipeline(Xtr, 3000 + s)[0] for s in range(5)]
sj = [jaccard(x["key"], y_["key"]) for x, y_ in combinations(seed_sets, 2)]
stab_rows.append({"setting": "full train, seeds only", "method": "Neural (verified)", "mean_jaccard": float(np.mean(sj)), "sd_jaccard": float(np.std(sj)),
                  "mean_n_rules": float(np.mean([len(x) for x in seed_sets])), "share_rules_in_90pct_runs": np.nan})
stab = pd.DataFrame(stab_rows)
stab.to_csv("results/e4_stability.csv", index=False)
S["e4"] = stab.to_dict("records")
fig, ax = plt.subplots(figsize=(6, 3.2))
w = .26
labs = ["full train", "n=2000"]
for i, (m, col) in enumerate([("Apriori", C_AP), ("Neural (verified)", C_NN), ("Neural consensus", C_CONS)]):
    v = [stab[(stab.setting == l) & (stab.method == m)]["mean_jaccard"].iloc[0] for l in labs]
    s_ = [stab[(stab.setting == l) & (stab.method == m)]["sd_jaccard"].iloc[0] for l in labs]
    ax.bar(np.arange(2) + (i - 1) * w, v, w, yerr=s_, color=col, label=m, capsize=2)
ax.set_xticks(range(2)); ax.set_xticklabels(["bootstrap, full train (n=34k)", "bootstrap, n=2000"])
ax.set_ylabel("mean pairwise Jaccard of rule sets"); ax.set_ylim(0, 1.25); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
ax.set_title("Rule-set stability (8 bootstrap resamples)")
save("fig_stability")

# ---------------------------------------------------------------- E5: spurious rules (permutation null + FDR + productivity)
print("E5 spurious rules", flush=True)
null_rows = []
for n in (1000, 5000):
    for rep in range(5):
        rs = np.random.default_rng(SEED + 77 + rep)
        sub_cat = cat_tr.iloc[rs.choice(len(cat_tr), n, replace=False)].reset_index(drop=True)
        perm = sub_cat.apply(lambda col: col.sample(frac=1, random_state=int(rs.integers(1e9))).values)   # destroy ALL dependencies
        Xp = D.to_onehot(perm, 0.01)
        a = MN.mine_apriori(Xp, MIN_SUP, MIN_CONF, MIN_LIFT)
        r, c, _ = neural_pipeline(Xp, SEED + rep)
        null_rows.append({"n": n, "rep": rep, "Apriori": len(a), "Neural (verified)": len(r), "Neural consensus": len(consensus(r))})
nul = pd.DataFrame(null_rows)
nul.to_csv("results/e5_permutation_null.csv", index=False)
S["e5_null"] = nul.groupby("n")[["Apriori", "Neural (verified)", "Neural consensus"]].mean().to_dict("index")


def fisher_fdr(r, Xs, alpha=0.05):
    st = MN.stats_on(Xs, r)
    n = len(Xs)
    nc = np.array([Xs[c].sum() for c in r["consequent"]])
    p = []
    for nac, na, ncc in zip(st["count"], st["ante_count"], nc):
        p.append(fisher_exact([[nac, na - nac], [ncc - nac, n - na - ncc + nac]], alternative="greater")[1])
    q = multipletests(p, method="fdr_bh")[1]
    return float((q < alpha).mean())


fdr_rows = []
rs = np.random.default_rng(SEED)
sub = Xtr.iloc[rs.choice(len(Xtr), 1000, replace=False)].reset_index(drop=True)
a1 = MN.mine_apriori(sub, MIN_SUP, MIN_CONF, MIN_LIFT)
r1, _, _ = neural_pipeline(sub, SEED)
for name, rr in [("Apriori", a1), ("Neural (verified)", r1), ("Neural consensus", consensus(r1))]:
    pr = MN.prune_unproductive(rr, sub)
    fdr_rows.append({"method": name, "n_rules": len(rr), "share_significant_BH_q05": fisher_fdr(rr, sub) if len(rr) else np.nan,
                     "share_productive": len(pr) / max(len(rr), 1)})
for name, rr in [("Apriori", ap), ("Neural (verified)", nn_rules), ("Neural consensus", cons_rules)]:
    pr = MN.prune_unproductive(rr, Xtr)
    fdr_rows.append({"method": name + " [full train]", "n_rules": len(rr), "share_significant_BH_q05": fisher_fdr(rr, Xtr),
                     "share_productive": len(pr) / len(rr)})
pd.DataFrame(fdr_rows).to_csv("results/e5_fdr_productivity.csv", index=False)
S["e5_fdr"] = fdr_rows

# ---------------------------------------------------------------- E6: understandability + explanations
print("E6 understandability", flush=True)
pruned = MN.prune_unproductive(cons_rules, Xtr)
und = []
for name, r in [("Apriori (raw)", ap), ("Neural consensus (raw)", cons_rules), ("Neural consensus + productivity pruning", pruned)]:
    L = r["antecedent"].map(len)
    und.append({"rule set": name, "n_rules": len(r), "mean_conditions": L.mean(), "share_<=2_conditions": float((L <= 2).mean()),
                "distinct_attributes_in_consequents": r["consequent"].map(D.attr_of).nunique()})
pd.DataFrame(und).to_csv("results/e6_understandability.csv", index=False)
S["e6"] = und
# rule cards (natural-language explanations with neural occlusion drivers)
top = pruned[pruned["consequent"].isin(["income=>50K", "income=<=50K"])].sort_values("lift", ascending=False)
# class rules for the high-income class come from the lower-support CAR mining of E7 -> computed below, so cards are built after E7
top = pd.concat([top[top.consequent == "income=<=50K"].head(2),
                 pruned[~pruned["consequent"].str.startswith("income")].sort_values("lift", ascending=False).head(3)])
cards = []
for _, r in top.iterrows():
    p_full, drv = NN.occlusion_attribution(ae, Xtr, r["antecedent"], r["consequent"])
    cards.append({"rule": r["key"], "support": r["support"], "confidence": r["confidence"], "lift": r["lift"], "count": int(r["count"]),
                  "network_P": p_full, "drivers": json.dumps({k: round(v, 3) for k, v in drv.items()}),
                  "explanation": EX.explain_rule(r, drivers=drv)})
cards = pd.DataFrame(cards)
cards.to_csv("results/e6_rule_cards.csv", index=False)
L = {n: r["antecedent"].map(len).value_counts().sort_index() for n, r in [("Apriori", ap), ("Neural consensus", cons_rules), ("Neural cons. + pruning", pruned)]}
pd.DataFrame(L).fillna(0).astype(int).to_csv("results/e6_rule_length_distribution.csv")

# ---------------------------------------------------------------- E7: downstream class-association-rule classifier + fairness
print("E7 downstream + fairness", flush=True)
TARGET = "income=>50K"
CAR_SUP, CAR_CONF, CAR_LIFT = 0.02, 0.60, 1.5


def car_apriori(Xs, drop_cols=()):
    """Class association rules (consequent = income>50K) from classical Apriori itemsets."""
    Z = Xs.drop(columns=list(drop_cols))
    fi = apriori(Z, min_support=CAR_SUP, use_colnames=True, max_len=4)
    sup = {frozenset(s): v for s, v in zip(fi["itemsets"], fi["support"])}
    base = Z[TARGET].mean()
    out = []
    for s, v in sup.items():
        if TARGET in s and len(s) > 1:
            a = s - {TARGET}
            conf = v / sup[a]
            if conf >= CAR_CONF and conf / base >= CAR_LIFT:
                out.append((tuple(sorted(a)), TARGET, v, conf, conf / base))
    return pd.DataFrame(out, columns=["antecedent", "consequent", "support", "confidence", "lift"])


def car_neural(Xs, drop_cols=()):
    Z = Xs.drop(columns=list(drop_cols))
    ae_ = NN.DenoisingAE(Z.shape[1], 64, 16, SEED)
    ae_.fit(Z.values, epochs=40)
    ante = NN.frequent_antecedents(Z, CAR_SUP, 3)
    items = list(Z.columns)
    V = np.zeros((len(ante), Z.shape[1]), np.float32)
    for i, a in enumerate(ante):
        V[i, list(a)] = 1
    P = ae_.forward(V)[:, items.index(TARGET)]
    M = Z.values
    base = M[:, items.index(TARGET)].mean()
    out = []
    for a, p in zip(ante, P):
        if items.index(TARGET) in a:
            continue
        if p >= CAR_CONF * 0.95 and p / base >= CAR_LIFT * 0.95:
            m = np.logical_and.reduce(M[:, list(a)], axis=1)
            na, nac = m.sum(), (m & M[:, items.index(TARGET)]).sum()
            conf = nac / na
            if nac / len(M) >= CAR_SUP and conf >= CAR_CONF and conf / base >= CAR_LIFT:
                out.append((tuple(sorted(items[i] for i in a)), TARGET, nac / len(M), conf, conf / base))
    return pd.DataFrame(out, columns=["antecedent", "consequent", "support", "confidence", "lift"])


def car_score(rules, Xs):
    M = Xs
    sc = np.zeros(len(M))
    for a, cf in zip(rules["antecedent"], rules["confidence"]):
        m = np.logical_and.reduce(M[list(a)].values, axis=1)
        sc = np.where(m, np.maximum(sc, cf), sc)
    return sc


def fairness(yt, yp, groups):
    res = {}
    for g in np.unique(groups):
        m = groups == g
        pos = m & (yt == 1); neg = m & (yt == 0)
        res[g] = {"sel": yp[m].mean(), "tpr": yp[pos].mean() if pos.sum() else np.nan, "fpr": yp[neg].mean() if neg.sum() else np.nan}
    sel = np.array([v["sel"] for v in res.values()]); tpr = np.array([v["tpr"] for v in res.values()])
    return {"DPD": sel.max() - sel.min(), "DIR": sel.min() / sel.max() if sel.max() > 0 else np.nan, "EOD": np.nanmax(tpr) - np.nanmin(tpr)}


sex_te = cat.iloc[idx_te]["sex"].values
race_te = cat.iloc[idx_te]["race"].values
sens_cols = [c for c in X.columns if D.attr_of(c) in D.SENSITIVE_ATTRS]
yte = Xte[TARGET].values.astype(int)
ytr = Xtr[TARGET].values.astype(int)
ds_rows = []
car_store = {}
NEG = "income=<=50K"                                  # complement of the target: must never be a feature (label leakage)
for variant, drop in [("with sex/race", [NEG]), ("without sex/race", sens_cols + [NEG])]:
    Ztr = Xtr.drop(columns=[TARGET] + list(drop)); Zte = Xte.drop(columns=[TARGET] + list(drop))
    models = {}
    t0 = time.time(); ra = car_apriori(Xtr, drop); ta = time.time() - t0
    t0 = time.time(); rn = car_neural(Xtr, drop); tn = time.time() - t0
    car_store[variant] = (ra, rn)
    Xtr_f, Xte_f = Xtr.drop(columns=list(drop)), Xte.drop(columns=list(drop))
    models["Apriori-CAR"] = (car_score(ra, Xte_f), len(ra), ta)
    models["Neural-CAR"] = (car_score(rn, Xte_f), len(rn), tn)
    dt = DecisionTreeClassifier(max_depth=5, random_state=SEED, class_weight="balanced").fit(Ztr, ytr)
    models["Decision tree (depth 5)"] = (dt.predict_proba(Zte)[:, 1], np.nan, np.nan)
    models["Majority class"] = (np.zeros(len(Zte)), 0, 0)
    for mname, (sc, nr, secs) in models.items():
        yp = (sc >= 0.5).astype(int) if mname == "Decision tree (depth 5)" else (sc > 0).astype(int)
        row = {"variant": variant, "model": mname, "n_rules": nr, "accuracy": accuracy_score(yte, yp), "balanced_acc": balanced_accuracy_score(yte, yp),
               "precision": precision_score(yte, yp, zero_division=0), "recall": recall_score(yte, yp), "F1": f1_score(yte, yp),
               "AUC": roc_auc_score(yte, sc) if sc.std() > 0 else 0.5, "seconds": secs}
        fs = fairness(yte, yp, sex_te); fr = fairness(yte, yp, race_te)
        row.update({"sex_DPD": fs["DPD"], "sex_DIR": fs["DIR"], "sex_EOD": fs["EOD"], "race_DPD": fr["DPD"], "race_DIR": fr["DIR"], "race_EOD": fr["EOD"]})
        ds_rows.append(row)
ds = pd.DataFrame(ds_rows)
ds.to_csv("results/e7_downstream_fairness.csv", index=False)
S["e7"] = ds.to_dict("records")
S["car_rule_counts"] = {v: {"apriori": len(a), "neural": len(n)} for v, (a, n) in car_store.items()}
S["car_overlap_jaccard"] = {v: jaccard([rule_key for rule_key in (MN.rule_key(x, TARGET) for x in a["antecedent"])],
                                       [rule_key for rule_key in (MN.rule_key(x, TARGET) for x in n["antecedent"])]) for v, (a, n) in car_store.items()}
# extra rule cards for the high-income class (class rules, no sensitive items), pruned for productivity
ra_, rn_ = car_store["without sex/race"]
ra_ = ra_.copy(); ra_["count"] = (ra_["support"] * len(Xtr)).round().astype(int)
Xns = Xtr.drop(columns=sens_cols + ["income=<=50K"])
ra_p = MN.prune_unproductive(ra_[ra_["lift"] > 0].reset_index(drop=True), Xns).sort_values("confidence", ascending=False)
S["car_pruned_rules"] = len(ra_p)
extra = []
for _, r_ in ra_p.head(4).iterrows():
    p_full, drv = NN.occlusion_attribution(ae, Xtr, r_["antecedent"], r_["consequent"])
    extra.append({"rule": MN.rule_key(r_["antecedent"], r_["consequent"]), "support": r_["support"], "confidence": r_["confidence"], "lift": r_["lift"],
                  "count": int(r_["count"]), "network_P": p_full, "drivers": json.dumps({k: round(v, 3) for k, v in drv.items()}),
                  "explanation": EX.explain_rule(r_, drivers=drv)})
cards = pd.concat([pd.DataFrame(extra), cards], ignore_index=True)
cards.to_csv("results/e6_rule_cards.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.0))
for ax_, (_, r_) in zip(axs, cards.iloc[:3].iterrows()):
    d = json.loads(r_["drivers"])
    ax_.barh([k.replace("=", ": ") for k in d], list(d.values()), color=C_NN)
    ax_.set_title(f"=> income >50K  (conf {r_['confidence']:.0%})", fontsize=9)
    ax_.set_xlabel("drop in network P(consequent)\nwhen item removed")
save("fig_occlusion")
# rule audit: sensitive items in mined rules and proxies
sens_in = ap["antecedent"].map(lambda a: any(D.attr_of(i) in D.SENSITIVE_ATTRS for i in a))
sens_cons = ap["consequent"].map(lambda c: D.attr_of(c) in D.SENSITIVE_ATTRS)
S["audit"] = {"n_rules": len(ap), "share_with_sensitive_antecedent": float(sens_in.mean()), "share_with_sensitive_consequent": float(sens_cons.mean()),
              "share_either": float((sens_in | sens_cons).mean())}
aud = ap[(ap["antecedent"].map(lambda a: any(D.attr_of(i) in D.SENSITIVE_ATTRS for i in a))) & ap["consequent"].str.startswith("income")]
aud = aud.sort_values("lift", ascending=False).head(8)[["key", "support", "confidence", "lift"]]
aud.to_csv("results/e7_sensitive_rule_audit.csv", index=False)
# base-rate gap in the data itself (what the rules are mirroring)
gap = cat.groupby("sex")["income"].apply(lambda s: (s == ">50K").mean()).to_dict()
gap_r = cat.groupby("race")["income"].apply(lambda s: (s == ">50K").mean()).to_dict()
S["base_rate_gap"] = {"sex": gap, "race": gap_r}
# proxy check: how well do non-sensitive items predict sex?  (rules like relationship=Husband => sex=Male)
prox = ap[ap["consequent"].isin(["sex=Male", "sex=Female"])].sort_values("confidence", ascending=False).head(5)[["key", "confidence", "lift"]]
prox.to_csv("results/e7_proxy_rules.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
for k, (title, cols_) in enumerate([("Accuracy / F1 / AUC", ["accuracy", "F1", "AUC"]), ("Disparate impact ratio (higher = fairer, 1 = parity)", ["sex_DIR", "race_DIR"])]):
    sub_ = ds[ds.variant == "with sex/race"]
    sub_ = sub_[sub_.model != "Majority class"]
    xs = np.arange(len(sub_)); w = .8 / len(cols_)
    for i, c in enumerate(cols_):
        ax[k].bar(xs + (i - (len(cols_) - 1) / 2) * w, sub_[c], w, label=c)
    ax[k].set_xticks(xs); ax[k].set_xticklabels([m.replace(" (depth 5)", "") for m in sub_["model"]], fontsize=8)
    ax[k].set_title(title, fontsize=9); ax[k].legend(frameon=False, fontsize=8)
save("fig_downstream_fairness")

# ---------------------------------------------------------------- E8: privacy-utility (noisy verification of proposals)
print("E8 privacy", flush=True)
cnt_ante = {}
Mtr = Xtr.values
col_ix = {c: i for i, c in enumerate(Xtr.columns)}
base_cnt = Mtr.sum(0)
cd = nn_cand.copy()
na_ = np.array([np.logical_and.reduce(Mtr[:, [col_ix[i] for i in a]], axis=1).sum() for a in cd["antecedent"]])
nac_ = cd["count"].values
nc_ = np.array([base_cnt[col_ix[c]] for c in cd["consequent"]])
n = len(Xtr)
priv = []
rng = np.random.default_rng(SEED)
for eps in (0.01, 0.05, 0.1, 0.5, 1.0, np.inf):
    res = []
    for rep in range(10):
        if np.isinf(eps):
            a_n, ac_n, c_n = na_, nac_, nc_
        else:
            sc = 3.0 / eps                                       # three count queries, eps/3 each (L1 sensitivity 1)
            a_n = na_ + rng.laplace(0, sc, len(na_)); ac_n = nac_ + rng.laplace(0, sc, len(na_)); c_n = nc_ + rng.laplace(0, sc, len(na_))
        conf = ac_n / np.maximum(a_n, 1); lift = conf / np.maximum(c_n / n, 1e-9); sup = ac_n / n
        keep = (sup >= MIN_SUP) & (conf >= MIN_CONF) & (lift >= MIN_LIFT)
        k = set(cd["key"][keep])
        res.append((len(k & K_ap) / max(len(k), 1), len(k & K_ap) / len(K_ap), len(k)))
    res = np.array(res)
    priv.append({"epsilon": "inf (no noise)" if np.isinf(eps) else eps, "precision": res[:, 0].mean(), "recall": res[:, 1].mean(), "n_rules": res[:, 2].mean()})
pr = pd.DataFrame(priv)
pr["F1"] = 2 * pr.precision * pr.recall / (pr.precision + pr.recall)
pr.to_csv("results/e8_privacy_utility.csv", index=False)
S["e8"] = pr.to_dict("records")
S["k_anonymity"] = {"min_rule_support_count": int(ap["count"].min()), "median_support_count": int(ap["count"].median())}
fig, ax = plt.subplots(figsize=(4.6, 3.2))
xs_ = [str(e) for e in pr["epsilon"]]
ax.plot(xs_, pr["precision"], "s--", label="precision", color=C_AP); ax.plot(xs_, pr["recall"], "o-", label="recall", color=C_NN)
ax.set_xlabel("epsilon (per released rule, 3 noisy counts)"); ax.set_ylabel("vs exact Apriori rules"); ax.legend(frameon=False)
ax.set_title("Privacy-utility trade-off (Laplace)")
plt.xticks(rotation=25)
save("fig_privacy")

# ---------------------------------------------------------------- workflow diagram
import workflow_fig
workflow_fig.draw("figures/fig_workflow.png")

S["total_runtime_min"] = (time.time() - t_start) / 60
json.dump(S, open("results/summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("done in %.1f min" % S["total_runtime_min"])
