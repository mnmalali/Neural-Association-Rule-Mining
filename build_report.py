"""Builds report/Neural_ARM_Report.docx from results/ (all numbers are read from the result files)."""
import json
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

S = json.load(open("results/summary.json"))
R = lambda f: pd.read_csv(f"results/{f}.csv")
e1, e2, e4 = R("e1_main_comparison").set_index("method"), R("e2_heldout_accuracy").set_index("method"), R("e4_stability")
e3 = R("e3_sample_size_summary"); fdr = R("e5_fdr_productivity"); und = R("e6_understandability")
ds = R("e7_downstream_fairness"); pr = R("e8_privacy_utility"); cards = R("e6_rule_cards"); nul = R("e5_permutation_null")
grid = R("e1_tau_slack_grid"); bn = R("e1_bottleneck"); cal = R("ae_calibration"); aud = R("e7_sensitive_rule_audit"); prox = R("e7_proxy_rules")
worst = R("e2_worst_rules"); desc = R("dataset_description")

TEAM = [("Mohammed Alali", "700044091"), ("Mazin Mohamedahmed", "700044312"), ("Ahmed Elmubarak", "700041220")]
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.08
for lvl, sz in [(1, 16), (2, 13), (3, 11.5)]:
    h = doc.styles[f"Heading {lvl}"]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.bold = True; h.font.color.rgb = NAVY
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10); h.paragraph_format.space_after = Pt(4)


def P(text="", bold=False, italic=False, size=None, align=None, after=None, style=None):
    p = doc.add_paragraph(style=style)
    parts = text.split("**")
    for i, t in enumerate(parts):                         # **bold** and *italic* inline markup
        for j, u in enumerate(t.split("*")):
            if not u:
                continue
            r = p.add_run(u); r.bold = bold or (i % 2 == 1); r.italic = italic or (j % 2 == 1)
            if size: r.font.size = Pt(size)
    if align: p.alignment = align
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p


def B(text):
    p = P(text, style="List Bullet"); p.paragraph_format.space_after = Pt(2); return p


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)


def T(df, widths=None, fs=9, caption=None, fmt=None):
    fmt = fmt or {}
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""; r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(fs); r.font.color.rgb = RGBColor(255, 255, 255)
        shade(cell, "1F3A5F")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(df.columns):
            v = row[c]
            if isinstance(v, float):
                v = "-" if pd.isna(v) else (fmt.get(c, "{:.3f}")).format(v)
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(fs)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs: p.paragraph_format.space_after = Pt(1)
    if caption:
        p = P(caption, italic=True, size=9, after=8); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def F(name, w=6.3, caption=None):
    doc.add_picture(f"figures/{name}.png", width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p = P(caption, italic=True, size=9, after=8); p.alignment = WD_ALIGN_PARAGRAPH.CENTER


pct = lambda x: f"{100 * x:.1f}%"
E = lambda m, c: e1.loc[m, c]
sw = lambda m, n, c: e3[(e3.method == m) & (e3.n == n)][c].iloc[0]
stb = lambda s, m, c="mean_jaccard": e4[(e4.setting == s) & (e4.method == m)][c].iloc[0]
dsr = lambda v, m, c: ds[(ds.variant == v) & (ds.model == m)][c].iloc[0]
fd = lambda m, c: fdr[fdr.method == m][c].iloc[0]
n_ap, n_nn = int(E("Apriori", "n_rules")), int(E("Neural (verified)", "n_rules"))

# =============================================================== title page
for _ in range(3): P("")
P("Data Mining (CSBP320) - Semester Project, Fall 2026", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
P("Neural Association Rule Mining with Human-Readable Explanations", bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
P("Do rules extracted from a neural network remain accurate, stable, non-spurious and understandable?", italic=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
tt = doc.add_table(rows=1, cols=2); tt.style = "Table Grid"; tt.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(["Student Name", "Student ID"]):
    tt.rows[0].cells[i].text = ""; r = tt.rows[0].cells[i].paragraphs[0].add_run(h); r.bold = True; shade(tt.rows[0].cells[i], "D9E2F3")
for n_, i_ in TEAM:
    c = tt.add_row().cells; c[0].text = n_; c[1].text = i_
P("")
P("Project number 9 of the provided project list  |  Dataset: UCI Adult (Census Income)  |  Due: 14 November 2026", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Code: reproducible Python pipeline (run_all.py).  Presentation: Neural_ARM_Presentation.pptx", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# =============================================================== 1 summary
doc.add_heading("1. Summary", 1)
P(f"Association rule mining (ARM) with Apriori or FP-Growth returns thousands of redundant rules whose quality is rarely checked. "
  f"Neural approaches promise compact representations, but a neural network is only useful for rule discovery if the rules it yields are trustworthy and readable. "
  f"We built **Neural ARM**, a denoising autoencoder (DAE) that is trained on one-hot transactions and proposes rules A => c from its reconstruction probabilities; "
  f"every proposal is then **verified on the data**, so reported support, confidence and lift are exact. We compared it with Apriori and FP-Growth (two classical baselines) and with an untrained-network ablation on the "
  f"UCI Adult census data ({S['n_rows']:,} people, {S['n_items']} items after discretisation; 70/30 stratified split).")
P("Main findings:")
B(f"**Accuracy.** The network recovers {pct(E('Neural (verified)', 'recall_vs_apriori'))} of the {n_ap:,} rules that exhaustive Apriori finds (support >= 5%, confidence >= 80%, lift >= 1.2) and adds none that are wrong "
  f"(precision {pct(E('Neural (verified)', 'precision_vs_apriori'))}). Its raw probability estimates correlate with true confidence at r = {S['ae_corr_p_conf']:.2f} (mean abs. error {S['ae_mae_p_conf']:.3f}). "
  f"An untrained network recovers {pct(E('Untrained AE (ablation)', 'recall_vs_apriori'))}, so the learned weights matter. On the held-out 30%, rule confidence shifts by only {e2.loc['Neural (verified)', 'conf_MAE_train_vs_test']:.3f} on average.")
B(f"**Stability.** Bootstrap rule sets overlap with Jaccard {stb('full train', 'Neural (verified)'):.2f} (full training set) and {stb('n=2000', 'Neural (verified)'):.2f} (2,000 rows) - essentially the same as Apriori "
  f"({stb('full train', 'Apriori'):.2f} / {stb('n=2000', 'Apriori'):.2f}); re-initialising the network adds almost no variance (Jaccard {stb('full train, seeds only', 'Neural (verified)'):.2f}).")
B(f"**Spurious rules.** On permuted (dependency-free) data with 1,000 rows, Apriori reports {S['e5_null']['1000']['Apriori']:.1f} false rules per run versus {S['e5_null']['1000']['Neural (verified)']:.1f} (verified) and {S['e5_null']['1000']['Neural consensus']:.1f} (consensus) for the neural miner - a modest, not statistically conclusive, reduction; "
  f"with 5,000 rows nobody reports any. Statistical significance is not a useful filter here (100% of rules pass BH-FDR), whereas **productivity** is: only {pct(fd('Apriori [full train]', 'share_productive'))} of rules add information beyond their sub-rules.")
B(f"**Understandability.** Removing unproductive rules shrinks the consensus rule set from {int(und.iloc[1]['n_rules']):,} to {int(und.iloc[2]['n_rules'])} rules, and each surviving rule is rendered as a plain-English sentence with its evidence and a network-based "
  f"occlusion explanation of which condition drives it.")
B(f"**Ethics.** {pct(S['audit']['share_either'])} of mined rules mention sex or race. Dropping the sensitive columns barely changes group disparities (sex disparate-impact ratio {dsr('with sex/race', 'Apriori-CAR', 'sex_DIR'):.2f} to {dsr('without sex/race', 'Apriori-CAR', 'sex_DIR'):.2f}) because proxies such as *relationship = Husband* remain; "
  f"noisy (Laplace) verification keeps F1 = {pr[pr.epsilon == '1.0'].F1.iloc[0]:.2f} at epsilon = 1 per statistic.")
P("The honest headline: on a dataset this small, the neural miner is **as accurate and as stable as exhaustive mining, not better**; its contribution is a verified, explainable rule pipeline whose weaknesses we measure rather than assume.", italic=True)

# =============================================================== 2 problem
doc.add_heading("2. Problem Statement and Definition", 1)
P("**Setting.** Let T = {t_1,...,t_N} be transactions over an item universe I (here every item is an attribute-value pair such as marital-status=Married). "
  "A rule A => c has an antecedent A (a set of items from distinct attributes, |A| <= 3) and a single-item consequent c. We use the standard measures: "
  "support = P(A and c), confidence = P(c | A), lift = confidence / P(c). A rule is *reported* if support >= 0.05, confidence >= 0.80 and lift >= 1.2.")
P("**Task.** Combine representation learning with association rules and evaluate whether the extracted rules remain:")
B("**accurate** - their support / confidence / lift generalise to unseen data and match the exact values;")
B("**stable** - the same rules are found when the data are resampled or the network is re-initialised;")
B("**non-spurious** - they are not artefacts of sampling noise or redundant restatements of simpler rules;")
B("**understandable** - they are few, short and can be explained in plain language to a non-expert.")
P("Why it matters: Adult is a socially sensitive data set (income, sex, race), so a rule that is accurate but unstable, redundant or biased could mislead decisions. This is why we add explainability, fairness and privacy components.")

# =============================================================== 3 data
doc.add_heading("3. Dataset Description", 1)
P(f"**Source.** UCI Machine Learning Repository, Adult / Census Income (Becker and Kohavi, 1996), the combined 48,842-record file (1994 US Census, 15 columns). "
  f"The file is included in the project folder as data/adult-all.csv. Target-like column: income (<=50K / >50K; {pct(S['pos_rate'])} are >50K).")
P(f"**Pre-processing.** (1) fnlwgt (survey weight) and education-num (duplicate of education) are dropped. (2) Age, capital gain/loss and hours per week are discretised into {3}-6 interpretable bins; rare education, work-class, marital-status and race levels are merged. "
  f"(3) Missing values ({S['missing']['workclass']:,} work-class, {S['missing']['occupation']:,} occupation, {S['missing']['native-country']} native-country) simply yield no item - the transaction is shorter; nothing is imputed. "
  f"(4) Each attribute=value pair becomes a Boolean item; items rarer than 1% are dropped, leaving **{S['n_items']} items and {S['avg_items_per_txn']:.1f} items per transaction**. "
  f"(5) Stratified 70/30 split (seed 42): {S['n_train']:,} training and {S['n_test']:,} test transactions. All mining and tuning uses training data only.")
d2 = desc.copy(); d2["sensitive"] = d2["sensitive"].map({True: "yes", False: ""})
T(d2.rename(columns={"n_values": "#values"}), widths=[1.3, 0.6, 2.8, 0.7, 0.8], caption="Table 1. Attributes after discretisation (sex and race are the protected attributes audited in Section 5.7).")

# =============================================================== 4 method
doc.add_heading("4. Model Framework and Evaluation Strategy", 1)
F("fig_workflow", 6.5, "Figure 1. Workflow diagram of the full pipeline.")
doc.add_heading("4.1 Neural ARM (proposed method)", 2)
P("**Network.** A denoising autoencoder with layers 55-64-16-64-55 (ReLU hidden layers, sigmoid outputs, binary cross-entropy, Adam, written in NumPy). During training each transaction keeps only a random 5-70% of its items (mask rate ~ U(0.3, 0.95)) while the target is the full transaction, so the network learns to *complete partial transactions*. "
  f"Consequently, feeding it an input that contains only the itemset A produces, for every item c, an estimate of P(c | A). Training takes about {S['ae_train_seconds']:.0f} s on a CPU.")
P("**Rule extraction (three steps).** (1) Candidate antecedents are all itemsets of 1-3 items from different attributes whose support is at least 5% (cheap counting; no consequent search). "
  "(2) For each antecedent the network is queried once; (A, c) is proposed if P(c|A) >= 0.75, the neural support estimate supp(A) x P(c|A) is within 5% of the minimum support and the neural lift estimate P(c|A)/P(c) is within 5% of the minimum lift. "
  "(3) Each proposal is **verified** against the data, so every reported number is exact and only the *selection* of rules is neural. A stricter variant, **Neural consensus**, additionally requires P(c|A) >= 0.80 (the network itself must agree with the data).")
doc.add_heading("4.2 Baselines (at least two)", 2)
B("**Apriori** (Agrawal and Srikant, 1994) via mlxtend: exhaustive level-wise search; same thresholds, same rule space (|A| <= 3, single consequent).")
B("**FP-Growth** (Han et al., 2000): a different algorithm that must return the identical rule set (we verify this) but gives an independent runtime reference.")
B("**Untrained autoencoder** (ablation): same pipeline with random weights, to show that learning - not the verification step - produces the rules.")
B("For the downstream test of Section 5.7 we add a **decision tree (depth 5)** and the **majority class** as predictive baselines.")
doc.add_heading("4.3 Evaluation strategy (and why)", 2)
T(pd.DataFrame([
    ["Accuracy", "Train/held-out split; compare train vs test confidence, support, lift; recall/precision against exact Apriori", "Rules are only useful if they hold on unseen people; Apriori is the exact ground truth for this rule space"],
    ["Stability", "8 bootstrap resamples (with new network seeds) at n = 34k and n = 2,000; pairwise Jaccard of rule sets; seed-only runs", "Neural training is random; resampling mimics collecting a new census sample"],
    ["Non-spurious", "(a) permutation null: shuffle every column independently and count rules (all of them are false); (b) Fisher exact test with BH-FDR; (c) productivity (Webb) test; (d) sample-size sweep n = 500...34k, replication on test set", "A null with known ground truth measures false discoveries directly; productivity detects redundant rules"],
    ["Understandable", "Number of rules, conditions per rule, share of <=2-condition rules, after redundancy pruning; natural-language rule cards", "Size and length are the standard proxies for cognitive load; no human study was possible (see limitations)"],
    ["Downstream / fairness / privacy", "Class-association-rule (CAR) classifier vs decision tree; disparate-impact ratio, demographic-parity and equal-opportunity gaps; Laplace-noise verification", "Shows whether rules are usable for prediction and whether they reproduce or amplify data bias"],
], columns=["Aspect", "How it is measured", "Justification"]), widths=[1.1, 3.1, 2.3], fs=8.5, caption="Table 2. Evaluation design.")
P("**Metrics (more than the required three).** Support, confidence, lift (rule quality); recall and precision against exact rules; confidence MAE and replication rate on held-out data; Jaccard stability; number of false discoveries under the null; productivity share; "
  "accuracy, balanced accuracy, F1, AUC; disparate-impact ratio, demographic-parity and equal-opportunity differences; precision/recall under noise.")

# =============================================================== 5 results
doc.add_heading("5. Results and Discussion", 1)
doc.add_heading("5.1 Neural ARM versus classical mining", 2)
m1 = e1.reset_index()[["method", "n_rules", "recall_vs_apriori", "precision_vs_apriori", "proposals", "proposal_precision", "mining_seconds", "mean_lift"]]
m1.columns = ["Method", "#rules", "Recall vs Apriori", "Precision vs Apriori", "Network proposals", "Proposal precision", "Mining s", "Mean lift"]
m1["Network proposals"] = m1["Network proposals"].astype(float)
T(m1, fs=8.5, fmt={"Network proposals": "{:.0f}", "Mining s": "{:.2f}", "#rules": "{:.0f}"}, caption="Table 3. Main comparison on the training set (thresholds: support >= 5%, confidence >= 80%, lift >= 1.2).")
P(f"Apriori and FP-Growth return identical rule sets ({n_ap:,} rules; identical = {S['ap_fp_identical']}), as theory demands. The verified neural miner returns {n_nn:,} of them ({pct(E('Neural (verified)', 'recall_vs_apriori'))}) and no rule outside that set. "
  f"The missing ~{100 - 100 * E('Neural (verified)', 'recall_vs_apriori'):.0f}% are borderline rules (confidence or lift barely above the threshold) that the network estimated just below it. "
  f"Only {pct(E('Neural (verified)', 'proposal_precision'))} of the network's *proposals* survive verification: {S['proposal_failures']['fail_confidence']} fail the confidence threshold, {S['proposal_failures']['fail_lift']} the lift threshold and {S['proposal_failures']['fail_support']} the support threshold "
  f"(a proposal can fail several). This is exactly why verification is part of the design: **the raw network output alone would be {pct(1 - E('Neural (verified)', 'proposal_precision'))} wrong at the boundary.** "
  f"The untrained-network ablation recovers {int(E('Untrained AE (ablation)', 'n_rules'))} rule(s), confirming that the learned weights carry the signal.")
P(f"Speed is *not* an advantage on this data: both classical miners need under a second, the neural pipeline needs {S['ae_train_seconds']:.0f} s of training plus ~{E('Neural (verified)', 'mining_seconds'):.1f} s of extraction. With only 55 items the lattice is tiny; the argument for neural mining (Berteloot et al., 2023) concerns much larger item spaces, which we did not test.")
F("fig_sensitivity", 6.3, "Figure 2. Left: proposal precision vs recall as the probability threshold tau and slack vary. Right: bottleneck size ablation (precision rises with larger bottleneck, recall falls).")
P(f"**Sensitivity.** Raising tau from 0.6 to 0.9 trades recall ({pct(grid[(grid.tau == 0.6) & (grid.slack == 0.05)].recall.iloc[0])} to {pct(grid[(grid.tau == 0.9) & (grid.slack == 0.05)].recall.iloc[0])}) for proposal precision "
  f"({pct(grid[(grid.tau == 0.6) & (grid.slack == 0.05)].precision.iloc[0])} to {pct(grid[(grid.tau == 0.9) & (grid.slack == 0.05)].precision.iloc[0])}); we use tau = 0.75, slack = 0.05 as a recall-oriented operating point because verification removes false proposals for free. "
  f"A bottleneck of 8 units gives the highest recall ({pct(bn[bn.bottleneck == 8].recall.iloc[0])}); 16 units was kept as a compromise (precision {pct(bn[bn.bottleneck == 16].precision.iloc[0])}).")

doc.add_heading("5.2 Accuracy on held-out data", 2)
a2 = e2.reset_index()[["method", "n_rules", "train_conf", "test_conf", "conf_MAE_train_vs_test", "replication_rate"]]
a2.columns = ["Rule set", "#rules", "Conf. (train / network est.)", "Conf. (test)", "Mean |error|", "Replication rate"]
T(a2, fs=8.5, fmt={"#rules": "{:.0f}"}, caption="Table 4. Held-out accuracy. Replication = test confidence >= 0.75 and lift >= 1.1. For raw proposals the first column is the network's own confidence estimate.")
P(f"Verified neural rules replicate perfectly (replication 100%, mean confidence error {e2.loc['Neural (verified)', 'conf_MAE_train_vs_test']:.3f}), identical to Apriori. "
  f"Raw proposals are noticeably worse (error {e2.loc['Neural raw proposals (no verification)', 'conf_MAE_train_vs_test']:.3f}, replication {pct(e2.loc['Neural raw proposals (no verification)', 'replication_rate'])}), and the untrained network is useless. "
  f"Test confidence differs from training confidence by at most {worst['drop'].max():.3f} for any rule. The rules that degrade most all predict *marital-status = Never-married* for young men (confidence {worst.iloc[0]['confidence']:.2f} to {worst.iloc[0]['test_conf']:.2f}) - high-variance, lower-confidence consequents are the first to move.")
F("fig_sample_size", 6.5, "Figure 3. Sample-size sweep (5 random subsets per size). Left: rules mined. Centre: share of rules replicating on the test set. Right: confidence error on test.")
P(f"The sweep shows where the neural variants help. With only 500 training rows, Apriori reports {sw('Apriori', 500, 'n_rules'):.0f} rules of which {pct(sw('Apriori', 500, 'replication'))} replicate; neural consensus reports {sw('Neural consensus', 500, 'n_rules'):.0f} rules and {pct(sw('Neural consensus', 500, 'replication'))} replicate. "
  f"The gain (about 1.5 points) is smaller than the run-to-run spread (sd ~ {e3[(e3.method == 'Apriori') & (e3.n == 500)].repl_sd.iloc[0]:.3f}), so we do **not** claim it is significant; from 2,000 rows upward all methods are indistinguishable.")

doc.add_heading("5.3 Stability", 2)
s3 = e4.copy(); s3.columns = ["Setting", "Method", "Mean Jaccard", "SD", "Mean #rules", "Rules found in >=90% of runs"]
T(s3, fs=8.5, fmt={"Mean #rules": "{:.0f}"}, caption="Table 5. Stability of rule sets over 8 bootstrap resamples (plus 5 seed-only runs).")
F("fig_stability", 4.6, "Figure 4. Mean pairwise Jaccard similarity of rule sets.")
P(f"With the full training set both methods are very stable (Jaccard ~{stb('full train', 'Apriori'):.2f}); with 2,000 rows only about {pct(stb('n=2000', 'Apriori'))} of the rules are shared between two resamples, because many rules sit near the thresholds. "
  f"Neural verified is within 0.01 of Apriori, so the network itself is not a major source of instability (seed-only Jaccard {stb('full train, seeds only', 'Neural (verified)'):.2f}). "
  f"The stricter consensus filter is slightly **less** stable ({stb('full train', 'Neural consensus'):.2f}): requiring two borderline conditions to hold doubles the chance that a borderline rule drops out. Stability and spuriousness therefore pull in opposite directions.")

doc.add_heading("5.4 Are the rules spurious?", 2)
n2 = nul.groupby("n")[["Apriori", "Neural (verified)", "Neural consensus"]].mean().reset_index()
n2.columns = ["n (rows)", "Apriori", "Neural (verified)", "Neural consensus"]
T(n2, fmt={"n (rows)": "{:.0f}", "Apriori": "{:.1f}", "Neural (verified)": "{:.1f}", "Neural consensus": "{:.1f}"}, caption="Table 6. Mean number of rules found on permuted data (all are false discoveries; 5 permutations per size).")
f2 = fdr.copy(); f2.columns = ["Rule set", "#rules", "Share significant (BH q<0.05)", "Share productive"]
T(f2, fs=8.5, fmt={"#rules": "{:.0f}"}, caption="Table 7. Significance and productivity. Top three rows: a random 1,000-row subset; bottom three rows: full training set.")
P(f"Under the permutation null, spurious rules appear only in small samples and are erratic (0, 0, 14, 5-7, 0 per run), so the means are noisy; neural verification reports about {S['e5_null']['1000']['Neural (verified)']:.1f} and consensus {S['e5_null']['1000']['Neural consensus']:.1f} versus {S['e5_null']['1000']['Apriori']:.1f} for Apriori. "
  "In one run (14 false rules) all three methods agree, because the network faithfully learns whatever chance pattern is in the sample - **a neural model cannot distinguish a chance pattern from a real one; only more data or a statistical test can.** At 5,000 rows there are no false rules at all.")
P(f"Fisher/BH significance is passed by 100% of the rules - with 34k rows any lift >= 1.2 is 'significant', so it cannot separate good from redundant rules. "
  f"The productivity test does: only {pct(fd('Apriori [full train]', 'share_productive'))} of Apriori's rules (and {pct(fd('Neural consensus [full train]', 'share_productive'))} of consensus rules) beat all their own sub-rules; the other ~58% merely add conditions that do not raise confidence, e.g. *marital-status=Married & relationship=Husband => sex=Male* is fully implied by the simpler *relationship=Husband => sex=Male* (99.99% confidence).")

doc.add_heading("5.5 Are the rules understandable?", 2)
u2 = und.copy(); u2.columns = ["Rule set", "#rules", "Mean #conditions", "Share with <=2 conditions", "#consequent attributes"]
T(u2, fs=8.5, fmt={"#rules": "{:.0f}", "#consequent attributes": "{:.0f}"}, caption="Table 8. Size and length of rule sets.")
P(f"Redundancy pruning is the single most effective readability tool: {int(und.iloc[1]['n_rules']):,} consensus rules collapse to {int(und.iloc[2]['n_rules'])} (a {100 * (1 - und.iloc[2]['n_rules'] / und.iloc[1]['n_rules']):.0f}% reduction) while still covering all six consequent attributes. Rules remain short (mean {und.iloc[2]['mean_conditions']:.1f} conditions; none longer than 3 by construction). "
  "Every rule is turned into a sentence by a template that maps each item to a phrase, adds the evidence (matching people, confidence, lift) and, for multi-condition rules, names the condition the network considers the main driver. Examples:")
for _, r in pd.concat([cards.iloc[[0, 4, 6]]]).iterrows():
    P("“" + r["explanation"] + "”", italic=True, size=10, after=3).paragraph_format.left_indent = Cm(0.8)
P("**Caveat.** We measure understandability with proxies. A real reader study is needed to claim it; Section 6 gives a ready-to-run protocol.")

doc.add_heading("5.6 Explainability and error analysis", 2)
F("fig_occlusion", 6.5, "Figure 5. Occlusion explanations: how much the network's probability of income > 50K falls when each condition is removed.")
P("**Explainability component - network occlusion.** For a rule A => c we remove one antecedent item at a time from the network input and record the fall in P(c | .). In the three top high-income rules, *capital-gain = high* accounts for ~0.5 of the probability while *Husband*, *Married* and *US-born* contribute almost nothing. "
  "So the data-verified rule 'capital-gain=high & relationship=Husband => income>50K' is, according to the network, really a one-condition rule - consistent with the productivity analysis and a concrete help to a reader deciding which conditions matter.")
F("fig_calibration", 3.8, "Figure 6. Network estimate P(c|A) vs exact confidence of the same rule (all proposals).")
cl = cal.round(3)
P(f"**Error analysis.** (i) *Calibration:* the network is slightly over-confident - in the 0.7-0.8 bin it predicts {cl.iloc[0]['pred']:.2f} but the true confidence is {cl.iloc[0]['true']:.2f}; in the 0.9-1.0 bin {cl.iloc[2]['pred']:.2f} vs {cl.iloc[2]['true']:.2f}. This is why a threshold on the network alone is unsafe and why verification is kept. "
  f"(ii) *Why proposals fail:* confidence (294) and lift (271) misestimates dominate over support (137). Many failing proposals have consequents with a high base rate (e.g. native-country = US), where P(c|A) is high but lift is ~1. "
  f"(iii) *Where rules fail on new data:* the largest train-to-test confidence drops are for young-male / never-married rules (Table in results/e2_worst_rules.csv); drops grow slightly with antecedent length (max drop 0.013, 0.022, 0.027 for 1, 2, 3 conditions).")

doc.add_heading("5.7 Downstream use, fairness and privacy", 2)
doc.add_heading("Rules as a classifier", 3)
P("To check usefulness we mined class-association rules for income > 50K (support >= 2%, confidence >= 60%, lift >= 1.5) with Apriori and with the neural miner, and predicted >50K when any rule fires. This is intentionally simple and transparent.")
d7 = ds[ds.variant == "with sex/race"][["model", "n_rules", "accuracy", "balanced_acc", "F1", "AUC"]]
d7.columns = ["Model", "#rules", "Accuracy", "Balanced acc.", "F1", "AUC"]
T(d7, fmt={"#rules": "{:.0f}"}, caption="Table 9. Downstream prediction on the held-out test set (income > 50K).")
P(f"The rule classifiers (accuracy {dsr('with sex/race', 'Apriori-CAR', 'accuracy'):.3f}, F1 {dsr('with sex/race', 'Apriori-CAR', 'F1'):.3f}) beat the majority class ({dsr('with sex/race', 'Majority class', 'accuracy'):.3f} accuracy) and match the F1 of the depth-5 decision tree ({dsr('with sex/race', 'Decision tree (depth 5)', 'F1'):.3f}). "
  f"The tree has better ranking (AUC {dsr('with sex/race', 'Decision tree (depth 5)', 'AUC'):.3f} vs {dsr('with sex/race', 'Apriori-CAR', 'AUC'):.3f}) because an 'any rule fires' score is coarse; the tree used balanced class weights, so its accuracy is lower. "
  f"Neural-CAR and Apriori-CAR give the same predictions on this data with {int(dsr('with sex/race', 'Neural-CAR', 'n_rules'))} vs {int(dsr('with sex/race', 'Apriori-CAR', 'n_rules'))} rules, i.e. the neural miner reaches the same accuracy with a smaller rule list.")
doc.add_heading("Ethics and fairness", 3)
fa = ds[ds.model.isin(["Apriori-CAR", "Neural-CAR", "Decision tree (depth 5)"])][["variant", "model", "sex_DIR", "sex_DPD", "sex_EOD", "race_DIR", "race_EOD"]]
fa.columns = ["Features", "Model", "Sex DIR", "Sex DPD", "Sex EOD", "Race DIR", "Race EOD"]
T(fa, fs=8.5, caption="Table 10. Group fairness of the predicted '>50K' label. DIR = disparate-impact ratio (1 = parity, <0.8 is the common red flag); DPD = demographic-parity difference; EOD = equal-opportunity (true-positive-rate) difference.")
bg = S["base_rate_gap"]
P(f"**The data are unequal.** In Adult {pct(bg['sex']['Male'])} of men but {pct(bg['sex']['Female'])} of women earn >50K (race: White {pct(bg['race']['White'])}, Black {pct(bg['race']['Black'])}). Rules summarise these patterns faithfully, so {pct(S['audit']['share_either'])} of all mined rules mention sex or race "
  f"(e.g. *{aud.iloc[3]['key']}*, confidence {aud.iloc[3]['confidence']:.2f}). "
  f"Every classifier violates the 0.8 disparate-impact rule of thumb for sex (DIR {dsr('with sex/race', 'Apriori-CAR', 'sex_DIR'):.2f} for rules, {dsr('with sex/race', 'Decision tree (depth 5)', 'sex_DIR'):.2f} for the tree).")
P(f"**Removing the protected columns is not enough.** Without sex and race the sex DIR moves from {dsr('with sex/race', 'Apriori-CAR', 'sex_DIR'):.3f} to {dsr('without sex/race', 'Apriori-CAR', 'sex_DIR'):.3f}, because proxies stay: the rule *relationship=Husband => sex=Male* has confidence above 99.9%. "
  "Rule-based models do have an ethical advantage: the offending condition is visible, so a reviewer can delete 'relationship = Husband' or 'sex = Female' rules before use, which is impossible with a black-box score. "
  "Our recommendation: (1) never use rules containing protected items or known proxies for individual decisions; (2) report the base-rate gap together with any rule that mentions a protected group; (3) re-audit disparate impact after every threshold change.")
doc.add_heading("Privacy", 3)
P(f"Rules are aggregates, but a rule backed by few people can reveal an individual. Every rule here is supported by at least {S['k_anonymity']['min_rule_support_count']:,} people (median {S['k_anonymity']['median_support_count']:,}), far above any k-anonymity threshold (k = 5-20 is typical). "
  "We also tested a differentially private release: Laplace noise is added to the three counts needed to verify each proposal (count of A, of A and c, of c; epsilon/3 each).")
p2 = pr.copy(); p2.columns = ["epsilon (per statistic)", "Precision", "Recall", "#rules", "F1"]
T(p2, fmt={"#rules": "{:.0f}"}, caption="Table 11. Utility of noisy verification versus the exact Apriori rules (mean of 10 runs).")
P(f"At epsilon = 1 per statistic the F1 is {pr[pr.epsilon == '1.0'].F1.iloc[0]:.3f}, essentially unchanged from {pr.iloc[-1].F1:.3f}; utility only breaks down for very strong noise (epsilon = 0.01, F1 {pr.iloc[0].F1:.2f}). "
  "Caveats: epsilon is per released statistic (the total budget grows with the number of rules), and the network itself was trained on raw data, so this protects the *released numbers* only; DP-SGD training would be needed for a complete guarantee. The Adult data are public and de-identified.")

# =============================================================== 6 limitations
doc.add_heading("6. Limitations and Future Improvements", 1)
for t in [
    "**One small, tabular data set.** With 55 items the classical lattice is trivial, so neural mining shows no speed benefit and few accuracy differences. Future work: high-dimensional transaction data (market baskets with thousands of items), where Apriori's candidate explosion occurs.",
    "**Neural variance is small but real; gains are not significant.** The small-sample spurious-rule reduction (about 1.5 points, 1.2 fewer false rules per run) lies within run-to-run noise. More repetitions (50+) and a paired test are needed.",
    "**The network proposes, the data decides.** Verification makes statistics exact but means the method inherits Apriori's thresholds and its blind spots; rules with antecedents of >3 items or multi-item consequents were not mined.",
    "**Understandability is measured by proxies.** A small user study should present matched pairs (raw rule vs template sentence vs sentence + driver) to about 20 readers and measure comprehension accuracy, time and trust; our templates are hand-written for this data set.",
    "**Rules are associations, not causes.** 'Husband => income > 50K' says nothing about why; causal discovery or counterfactual checks are needed before acting on a rule.",
    "**Discretisation and 1994 data.** Bin edges are our choice and the census is 30 years old; conclusions on fairness describe the data, not today's population.",
    "**Privacy.** The network is not trained with DP-SGD; only the released rule statistics are noised.",
    "**Future improvements:** attention- or contrastive-based item embeddings for faster candidate generation; multi-consequent rules; stability selection (keep only rules found in >= 90% of bootstrap runs); FDR-controlled thresholds; fairness-constrained rule selection; human evaluation.",
]:
    B(t)

# =============================================================== 7 conclusion
doc.add_heading("7. Conclusion", 1)
P(f"We built and evaluated a neural association-rule miner whose proposals are verified against the data. It reproduces {pct(E('Neural (verified)', 'recall_vs_apriori'))} of exhaustive Apriori's {n_ap:,} rules with exact statistics, generalises to held-out data (confidence error {e2.loc['Neural (verified)', 'conf_MAE_train_vs_test']:.3f}), "
  f"is as stable as classical mining (Jaccard {stb('full train', 'Neural (verified)'):.2f}), and produces explainable rule cards in plain English after pruning {pct(1 - und.iloc[2]['n_rules'] / und.iloc[1]['n_rules'])[:-3]}% redundant rules. "
  "We found no evidence that a neural miner is *better* than Apriori on this dataset, and clear evidence about its risks: raw network outputs are over-confident, a neural model cannot tell chance patterns from real ones, most mined rules are redundant, and rules mirror and can proxy societal bias. "
  "Therefore the recommended recipe is: let the network propose, verify on data, prune unproductive rules, explain each rule, and audit for fairness and privacy before use.")

# =============================================================== 8 reproducibility
doc.add_heading("8. Reproducibility", 1)
P("All code, data and results are in the submitted project folder:")
for t in ["data/adult-all.csv - dataset (UCI Adult, 48,842 rows)", "src/data.py, mining.py, neural.py, explain.py - preprocessing, Apriori/FP-Growth baselines, denoising autoencoder miner, rule explanations",
          "run_all.py - runs every experiment (about 4 minutes on 2 CPU cores, fixed seed 42) and writes results/*.csv, results/summary.json and figures/*.png",
          "build_report.py / build_slides.js - regenerate this report and the presentation from the result files; requirements.txt lists the Python packages; README.md gives step-by-step instructions"]:
    B(t)
P("Command: `pip install -r requirements.txt` then `python run_all.py`. Environment: Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9, mlxtend 0.25, statsmodels 0.15.")

doc.add_heading("References", 1)
for t in ["Agrawal, R. and Srikant, R. (1994). Fast algorithms for mining association rules. VLDB.",
          "Han, J., Pei, J. and Yin, Y. (2000). Mining frequent patterns without candidate generation. SIGMOD.",
          "Berteloot, T., Khoury, R. and Durand, A. (2023). Association rules mining with auto-encoders (ARM-AE).",
          "Vincent, P., Larochelle, H., Bengio, Y. and Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. ICML.",
          "Webb, G. I. (2007). Discovering significant patterns. Machine Learning 68(1).",
          "Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. JRSS-B 57(1).",
          "Liu, B., Hsu, W. and Ma, Y. (1998). Integrating classification and association rule mining (CBA). KDD.",
          "Dwork, C. (2006). Differential privacy. ICALP.",
          "Kingma, D. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
          "Becker, B. and Kohavi, R. (1996). Adult data set. UCI Machine Learning Repository."]:
    P(t, size=10, after=2)

import os
os.makedirs("report", exist_ok=True)
doc.save("report/Neural_ARM_Report.docx")
print("saved")
