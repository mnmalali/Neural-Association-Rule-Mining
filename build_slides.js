// Builds report/Neural_ARM_Presentation.pptx from results/slide_numbers.json and figures/*.png
// usage: node build_slides.js   (needs pptxgenjs; applyTheme from the pptx skill if available)
const fs = require("fs");
const pptxgen = require("pptxgenjs");
let applyTheme = null;
try {
  const dir = process.env.PPTX_SKILL_DIR;
  if (dir) applyTheme = require(dir + "/scripts/apply_theme.js").applyTheme;
} catch (e) { /* theme patch is optional */ }

const N = JSON.parse(fs.readFileSync("results/slide_numbers.json", "utf8").replace(/\bNaN\b/g, "null"));
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const E1 = N.e1, E2 = N.e2;
const dsr = (v, m, c) => N.ds.find((r) => r.variant === v && r.model === m)[c];
const stab = (s, m) => N.stab.find((r) => r.setting === s && r.method === m).mean_jaccard;
const fdr = (m) => N.fdr.find((r) => r.method === m);

const THEME = {
  name: "NeuralARM", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2A41", lt1: "FFFFFF", dk2: "0B1F3A", lt2: "EEF3F8", accent1: "1F7A8C", accent2: "F2A541", accent3: "D1495B",
            accent4: "5B8E7D", accent5: "4A6FA5", accent6: "8D99AE", hlink: "1F7A8C", folHlink: "8D99AE" },
};
const HEX = { teal: "1F7A8C", amber: "F2A541", red: "D1495B", grey: "8D99AE", navy: "0B1F3A", green: "5B8E7D", blue: "4A6FA5" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Neural Association Rule Mining with Human-Readable Explanations";
pres.author = "Mohammed Alali, Mazin Mohamedahmed, Ahmed Elmubarak";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.75, fontSize: 28, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Neural ARM | Data Mining CSBP320", options: { x: 0.5, y: 5.22, w: 5, h: 0.25, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.22, w: 0.5, h: 0.25, fontSize: 10, color: C.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.28, w: 8.8, h: 0.75, fontSize: 28, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});

let slideNo = 0;
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
const card = (s, x, y, w, h, head, body, opt = {}) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opt.fill || C.background2 }, line: { color: opt.fill || C.background2 }, objectName: "card " + head });
  s.addText(head, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.36, fontSize: opt.hs || 15, bold: true, color: opt.hc || C.text2, margin: 0, isTextBox: true, valign: "top" });
  s.addText(body, { x: x + 0.15, y: y + 0.48, w: w - 0.3, h: h - 0.58, fontSize: opt.bs || 13, color: opt.bc || C.text1, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 3 });
};
const stat = (s, x, y, w, big, label, color) => {
  s.addText(big, { x, y, w, h: 0.62, fontSize: 34, bold: true, fontFace: "Cambria", color: color || C.accent1, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(label, { x, y: y + 0.62, w, h: 0.5, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
};
const chartBase = (title) => ({
  showTitle: !!title, title, titleFontSize: 13, titleColor: HEX.navy, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelColor: "1B2A41", valAxisLabelColor: "1B2A41", valGridLine: { color: "D9E0E8", size: 0.5 }, catGridLine: { style: "none" },
  dataLabelFontSize: 11, dataLabelColor: "1B2A41", legendFontSize: 11, legendColor: "1B2A41", catAxisLabelFontFace: "Calibri", valAxisLabelFontFace: "Calibri",
});
const bullets = (arr, o = {}) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1, paraSpaceAfter: 5, ...o } }));

// ============================================================ 1 title
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("", { placeholder: "title" });
  s.addText("Neural Association Rule Mining with Human-Readable Explanations", { x: 0.6, y: 0.9, w: 8.8, h: 1.5, fontSize: 36, bold: true, fontFace: "Cambria", color: C.background1, margin: 0, isTextBox: true, valign: "top" });
  s.addText("Are rules extracted from a neural network accurate, stable, non-spurious and understandable?", { x: 0.6, y: 2.5, w: 8.2, h: 0.7, fontSize: 18, italic: true, color: C.accent2, margin: 0, isTextBox: true, valign: "top" });
  s.addText([
    { text: "Mohammed Alali  700044091", options: { breakLine: true } },
    { text: "Mazin Mohamedahmed  700044312", options: { breakLine: true } },
    { text: "Ahmed Elmubarak  700041220" },
  ], { x: 0.6, y: 3.5, w: 5, h: 1.0, fontSize: 15, color: C.background1, margin: 0, isTextBox: true, valign: "top" });
  s.addText("Data Mining (CSBP320)  |  Semester Project, Fall 2026\nDataset: UCI Adult (Census Income)", { x: 5.6, y: 3.5, w: 3.8, h: 1.0, fontSize: 13, color: C.accent6, margin: 0, isTextBox: true, valign: "top", align: "right" });
  s.addNotes("Introduce the team and the question: neural networks can generate association rules, but are those rules trustworthy? We test four properties: accuracy, stability, non-spuriousness, understandability.");
}

// ============================================================ 2 problem
{
  const s = content("1. Problem statement and definition",
    "Rule A=>c: antecedent up to 3 items from different attributes, single consequent. Support, confidence, lift defined as usual. Thresholds are the same for all methods so the comparison is fair. The four properties on the right are exactly the four things the project brief asks us to evaluate.");
  s.addText([
    { text: "Transactions: each person = a set of attribute=value items", options: { bullet: true, breakLine: true } },
    { text: "Rule A => c, e.g. Married & Bachelors => income >50K", options: { bullet: true, breakLine: true } },
    { text: "support = P(A,c)   confidence = P(c|A)   lift = P(c|A) / P(c)", options: { bullet: true, breakLine: true } },
    { text: "Reported if support >= 5%, confidence >= 80%, lift >= 1.2", options: { bullet: true, breakLine: true } },
    { text: "Task: combine representation learning (autoencoder) with association rules, then test whether the rules stay trustworthy", options: { bullet: true, bold: true } },
  ], { x: 0.5, y: 1.2, w: 4.3, h: 3.7, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 7 });
  const items = [["Accurate", "Holds on unseen data; matches exact support / confidence / lift"], ["Stable", "Same rules after resampling data or re-seeding the network"],
                 ["Non-spurious", "Not chance patterns, not redundant restatements"], ["Understandable", "Few, short, explained in plain language"]];
  items.forEach((it, i) => {
    const x = 5.1 + (i % 2) * 2.2, y = 1.2 + Math.floor(i / 2) * 1.9;
    card(s, x, y, 2.1, 1.75, it[0], it[1], { hs: 15, bs: 12 });
  });
}

// ============================================================ 3 dataset
{
  const s = content("2. Dataset: UCI Adult (Census Income)",
    "Public, de-identified 1994 US census extract, 48,842 records. We drop fnlwgt and education-num, bin numeric columns, merge rare categories. Missing values simply produce no item; nothing is imputed. Items rarer than 1% are dropped. Sex and race are kept on purpose so that we can audit fairness later. Train/test split 70/30 stratified, seed 42.");
  stat(s, 0.5, 1.2, 2.1, N.n_rows.toLocaleString(), "people (34,189 train / 14,653 test)");
  stat(s, 2.75, 1.2, 2.1, String(N.n_items), "Boolean items (attribute=value)");
  stat(s, 5.0, 1.2, 2.1, N.ntxn.toFixed(1), "items per transaction", HEX.blue);
  stat(s, 7.25, 1.2, 2.2, pct(N.pos), "earn >50K (class imbalance)", HEX.amber);
  card(s, 0.5, 2.75, 4.35, 2.25, "Pre-processing", "Age, capital gain/loss, hours/week binned into 2-6 readable levels\nRare levels merged (education, work-class, marital, race)\nMissing work-class / occupation / country: no item", { bs: 13 });
  card(s, 5.05, 2.75, 4.4, 2.25, "Why this dataset", "Real, mixed-type, classic for ARM\nHas protected attributes (sex, race) for the ethics component\nSource: UCI repository (Becker and Kohavi, 1996); file shipped in data/", { bs: 13 });
}

// ============================================================ 4 framework
{
  const s = content("3. Model framework (workflow)", "Walk through left to right: data -> one-hot transactions -> split. The denoising autoencoder proposes rules, the baselines mine the same rule space exhaustively. Both go through exact verification on the data, then evaluation (accuracy, stability, spuriousness, understandability) and explanation/audit.");
  s.addImage({ path: "figures/fig_workflow.png", x: 0.5, y: 1.15, w: 9.0, h: 3.17, altText: "Workflow diagram: data, preprocessing, split, neural ARM and baselines, verification, evaluation, explanation and audit" });
  s.addText("The network proposes rules; the data verifies them; every rule is then evaluated, explained and audited.", { x: 0.5, y: 4.5, w: 9, h: 0.5, fontSize: 15, bold: true, color: C.accent1, margin: 0, isTextBox: true });
}

// ============================================================ 5 method
{
  const s = content("4. Neural ARM: rules from an autoencoder",
    "Key idea: train with heavy masking so the network learns to complete a partial transaction. Then feeding only antecedent A yields P(c|A) for every item. Because raw network estimates are slightly over-confident (see calibration), every proposal is verified on the data - the reported numbers are exact. Consensus = additionally require the network itself to give at least 0.80.");
  const steps = [["1  Train", "Denoising autoencoder 55-64-16-64-55. Each person keeps a random 5-70% of items; target = full transaction. 8 s on CPU."],
                 ["2  Propose", "Query with only antecedent A (1-3 items, support >= 5%). Propose A => c if P(c|A) >= 0.75 and estimated support and lift are near threshold."],
                 ["3  Verify", "Recompute support, confidence, lift on the data. Keep only rules that pass the thresholds. Exact statistics."]];
  steps.forEach((st, i) => card(s, 0.5 + i * 3.05, 1.2, 2.9, 2.35, st[0], st[1], { hs: 17, bs: 15 }));
  s.addText("Network output for input {A}  ~  P(c | A)", { x: 0.5, y: 3.75, w: 9, h: 0.4, fontSize: 18, bold: true, fontFace: "Cambria", color: C.accent1, margin: 0, isTextBox: true });
  s.addText("Variant 'Neural consensus' also requires P(c|A) >= 0.80, so the network and the data must agree.", { x: 0.5, y: 4.2, w: 9, h: 0.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
}

// ============================================================ 6 baselines
{
  const s = content("5. Baselines and ablation (3 methods + 2 predictive)",
    "Apriori and FP-Growth are different algorithms that provably return the same rule set; we verified this on our data (identical 1,214 rules), so FP-Growth is mainly a runtime check. The untrained autoencoder is the key ablation: same pipeline with random weights, to show that the learning, not the verification, produces the rules. For the downstream task we add a depth-5 decision tree and the majority class.");
  card(s, 0.5, 1.2, 4.35, 1.75, "Apriori", "Exhaustive level-wise search (mlxtend). Same thresholds and same rule space: the exact ground truth.", { bs: 13 });
  card(s, 5.05, 1.2, 4.4, 1.75, "FP-Growth", "Pattern-tree algorithm. Returns identical rules (checked: " + E1["Apriori"].n_rules + " = " + E1["FP-Growth"].n_rules + ").", { bs: 13 });
  card(s, 0.5, 3.1, 4.35, 1.85, "Untrained autoencoder (ablation)", "Random weights, same extraction + verification. Shows learning is what finds rules.", { bs: 13 });
  card(s, 5.05, 3.1, 4.4, 1.85, "Predictive baselines (Section 9)", "Decision tree (depth 5) and majority class, compared with rule-list classifiers.", { bs: 13 });
}

// ============================================================ 7 evaluation
{
  const s = content("6. Evaluation strategy and metrics",
    "We chose each test because it has a known ground truth: exact Apriori for accuracy, permuted data (all rules are false) for spuriousness, bootstrap for stability. Significance tests alone are not enough with 34k rows, so we add Webb's productivity test. We list well over three metrics.");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy }, fontSize: 12 } });
  const cell = (t, o = {}) => ({ text: t, options: { fontSize: 11, color: "1B2A41", ...o } });
  const rows = [
    [H("Aspect"), H("How it is tested (justification)"), H("Metrics")],
    [cell("Accuracy", { bold: true }), cell("70/30 split; compare with exact Apriori rules"), cell("recall, precision, confidence error, replication rate")],
    [cell("Stability", { bold: true }), cell("8 bootstrap resamples, new seeds; n = 34k and 2,000"), cell("pairwise Jaccard, rules found in >=90% of runs")],
    [cell("Non-spurious", { bold: true }), cell("Permutation null (all rules false); Fisher + BH-FDR; productivity; sample-size sweep"), cell("false rules, share significant, share productive")],
    [cell("Understandable", { bold: true }), cell("Rule count / length after pruning; natural-language rule cards"), cell("#rules, conditions per rule")],
    [cell("Use, fairness, privacy", { bold: true }), cell("Rule-list classifier vs decision tree; group audit; Laplace noise"), cell("accuracy, F1, AUC, DIR, DPD, EOD, epsilon")],
  ];
  s.addTable(rows, { x: 0.5, y: 1.2, w: 9.0, colW: [1.7, 4.4, 2.9], border: { type: "solid", color: "C9D3DE", pt: 0.75 }, fill: { color: "FFFFFF" }, rowH: [0.38, 0.55, 0.55, 0.85, 0.55, 0.7], valign: "middle", margin: 0.06 });
}

// ============================================================ 8 accuracy
{
  const s = content("7a. Results: accurate and exact after verification",
    `Neural verified recovers ${pct(E1["Neural (verified)"].recall_vs_apriori)} of Apriori's ${E1["Apriori"].n_rules} rules and adds no wrong ones. Only ${pct(E1["Neural (verified)"].proposal_precision)} of raw proposals survive verification - the raw network is over-confident near thresholds, which is why verification is part of the method. Untrained network: ${pct(E1["Untrained AE (ablation)"].recall_vs_apriori)}. Speed is not an advantage on 55 items: Apriori takes about half a second.`);
  s.addChart(pres.charts.BAR, [{ name: "Recall of exact Apriori rules", labels: ["Neural (verified)", "Neural consensus", "Untrained AE"], values: [E1["Neural (verified)"].recall_vs_apriori * 100, E1["Neural consensus"].recall_vs_apriori * 100, E1["Untrained AE (ablation)"].recall_vs_apriori * 100] }],
    { x: 0.5, y: 1.15, w: 5.0, h: 3.9, barDir: "col", chartColors: [HEX.teal], showLegend: false, showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", valAxisMaxVal: 110, valAxisMinVal: 0, valAxisLabelFormatCode: "0", ...chartBase("Recall of exact Apriori rules (%)") });
  stat(s, 5.9, 1.2, 3.5, pct(E2["Neural (verified)"].replication_rate, 0), "of verified rules replicate on held-out test data");
  stat(s, 5.9, 2.4, 3.5, E2["Neural (verified)"].conf_MAE_train_vs_test.toFixed(3), "mean |train - test| confidence error", HEX.blue);
  stat(s, 5.9, 3.6, 3.5, "r = " + N.corr.toFixed(2), "network P(c|A) vs true confidence; raw proposals only " + pct(E1["Neural (verified)"].proposal_precision, 0) + " precise", HEX.amber);
}

// ============================================================ 9 sample-size
{
  const s = content("7b. Small samples: only a slight neural gain",
    "Sweep over training size with 5 random subsets each. With 500 rows consensus replicates 96.3% versus 94.8% for Apriori, but the spread between runs is about 2 points, so we do not claim significance. From 2,000 rows all methods are indistinguishable. Neural mining also finds fewer rules at small n, i.e. it is slightly more conservative.");
  s.addImage({ path: "figures/fig_sample_size.png", x: 0.5, y: 1.15, w: 9.0, h: 2.62, altText: "Sample-size sweep: rules mined, held-out replication rate, confidence error for Apriori, neural, consensus" });
  s.addText(bullets([
    "n = 500: replication 94.8% (Apriori) vs 95.8% (neural) vs 96.3% (consensus), within run-to-run noise",
    "n >= 2,000: all methods indistinguishable; 100% replication from 5,000 rows",
  ], { fontSize: 14 }), { x: 0.5, y: 3.95, w: 9.0, h: 1.1, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ============================================================ 10 stability
{
  const s = content("7c. Stability: as stable as Apriori",
    "Eight bootstrap resamples, each with a different network seed; we compare all pairs of rule sets. With the full training set Jaccard is 0.94 for Apriori and 0.94 neural. At n=2000 about 77% overlap, mostly borderline rules. Re-seeding the network on identical data gives Jaccard 0.98, so the network adds little variance. Consensus is slightly less stable because two borderline conditions must both hold.");
  const labs = ["Bootstrap, full train", "Bootstrap, n = 2,000"];
  const g = (m) => [stab("full train", m), stab("n=2000", m)];
  s.addChart(pres.charts.BAR, [{ name: "Apriori", labels: labs, values: g("Apriori") }, { name: "Neural (verified)", labels: labs, values: g("Neural (verified)") }, { name: "Neural consensus", labels: labs, values: g("Neural consensus") }],
    { x: 0.5, y: 1.15, w: 5.6, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [HEX.red, HEX.teal, HEX.green], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", valAxisMaxVal: 1.1, valAxisMinVal: 0, valAxisLabelFormatCode: "0.0", ...chartBase("Mean pairwise Jaccard of rule sets") });
  stat(s, 6.5, 1.25, 3.0, stab("full train, seeds only", "Neural (verified)").toFixed(2), "Jaccard when only the network seed changes (same data)");
  s.addText("Stability and spuriousness pull in opposite directions: the stricter consensus filter is a little less stable (" + stab("full train", "Neural consensus").toFixed(2) + ").", { x: 6.5, y: 2.75, w: 3.0, h: 1.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ============================================================ 11 spurious
{
  const s = content("7d. Non-spurious? Significance is not enough",
    "On permuted data every dependency is destroyed, so any rule found is a false discovery. At 1,000 rows Apriori reports 5.2 false rules per run on average, neural 4.0, consensus 3.8 - but the counts are erratic (0, 0, 14, 5-7, 0) so this is not conclusive; in the 14-rule run all methods agree because the network learns whatever pattern is in the sample. At 5,000 rows nobody finds any. Fisher/BH significance is passed by 100% of rules with 34k rows, so it cannot separate good from redundant rules. Productivity (Webb): a rule is productive if it beats all of its sub-rules; only 42% do.");
  s.addChart(pres.charts.BAR, [{ name: "False rules per run", labels: ["Apriori", "Neural (verified)", "Neural consensus"], values: [N.null["1000"]["Apriori"], N.null["1000"]["Neural (verified)"], N.null["1000"]["Neural consensus"]] }],
    { x: 0.5, y: 1.15, w: 4.6, h: 3.9, barDir: "col", chartColors: [HEX.red], showLegend: false, showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: 7, valAxisLabelFormatCode: "0", ...chartBase("Rules found on permuted data, n = 1,000") });
  stat(s, 5.5, 1.2, 4.0, "0", "false rules at n = 5,000 for every method");
  stat(s, 5.5, 2.3, 4.0, "100%", "of rules pass Fisher + BH-FDR: significance cannot discriminate", HEX.blue);
  stat(s, 5.5, 3.4, 4.0, pct(fdr("Apriori [full train]").share_productive, 0), "of rules are productive; the rest only restate simpler rules", HEX.amber);
}

// ============================================================ 12 understandability
{
  const s = content("7e. Understandable: prune, then explain",
    `Productivity pruning cuts ${N.und[1].n_rules} consensus rules to ${N.und[2].n_rules} (59% fewer) and still covers all six consequent attributes. Each rule becomes a sentence: conditions, then confidence, then how many people it covers, then lift. Understandability is measured by proxies here (size, length); a user study is future work with a ready protocol in the report.`);
  const pick = [0, 4, 6];
  const heads = ["High income", "Low income", "Age pattern"];
  pick.forEach((idx, i) => {
    const c = N.cards[idx];
    card(s, 0.5, 1.2 + i * 1.28, 6.2, 1.18, heads[i] + "  (confidence " + pct(c.confidence, 0) + ", lift " + c.lift.toFixed(1) + ")", c.explanation.split(" Strongest")[0], { hs: 12, bs: 11 });
  });
  stat(s, 7.0, 1.2, 2.5, "1,172 → 479", "rules after removing unproductive ones (-59%)");
  stat(s, 7.0, 2.5, 2.5, "2.6", "conditions per rule on average (max 3)", HEX.blue);
  stat(s, 7.0, 3.7, 2.5, "32%", "of rules need only 1-2 conditions", HEX.amber);
}

// ============================================================ 13 explainability
{
  const s = content("8. Explainability and error analysis",
    "Occlusion: remove each antecedent item and see how much the network's probability for the consequent falls. In the top income rules 'capital gain = high' carries about 0.5 of the probability; Husband, Married, US-born add almost nothing - so these multi-condition rules are really one-condition rules. Calibration: the network is slightly over-confident (0.78 predicted vs 0.71 true in the 0.7-0.8 bin). Proposal failures are dominated by confidence and lift misestimates, often for high-base-rate consequents such as native-country = US. Largest train-to-test drops are for young / never-married rules.");
  s.addImage({ path: "figures/fig_occlusion.png", x: 0.5, y: 1.15, w: 9.0, h: 2.25, altText: "Occlusion explanations: capital gain high drives income above 50K rules" });
  s.addImage({ path: "figures/fig_calibration.png", x: 0.5, y: 3.5, w: 1.9, h: 1.55, altText: "Calibration of network estimate" });
  s.addText(bullets([
    "Occlusion: 'capital-gain = high' drives ~0.5 of P(income >50K); other conditions add ~0",
    "Calibration: network is slightly over-confident (0.78 predicted vs 0.71 true at 0.7-0.8)",
    "Failures: " + N.fail.fail_confidence + " confidence, " + N.fail.fail_lift + " lift, " + N.fail.fail_support + " support misestimates",
  ], { fontSize: 13 }), { x: 2.6, y: 3.55, w: 6.9, h: 1.5, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ============================================================ 14 downstream
{
  const s = content("9a. Are the rules useful? Rule list vs decision tree",
    "Class-association rules for income>50K (support>=2%, confidence>=60%, lift>=1.5); the classifier predicts >50K when any rule fires. Both rule classifiers beat the majority class and match the tree's F1. The tree has better AUC because 'any rule fires' is a coarse score; it was trained with balanced class weights so its accuracy is lower. Neural-CAR reaches the same predictions with fewer rules (109 vs 139).");
  const models = ["Apriori-CAR", "Neural-CAR", "Decision tree (depth 5)", "Majority class"];
  const v = "with sex/race";
  s.addChart(pres.charts.BAR, [
    { name: "Accuracy", labels: ["Apriori rules", "Neural rules", "Decision tree", "Majority"], values: models.map((m) => dsr(v, m, "accuracy")) },
    { name: "F1", labels: ["Apriori rules", "Neural rules", "Decision tree", "Majority"], values: models.map((m) => dsr(v, m, "F1")) },
    { name: "AUC", labels: ["Apriori rules", "Neural rules", "Decision tree", "Majority"], values: models.map((m) => dsr(v, m, "AUC")) }],
    { x: 0.5, y: 1.15, w: 6.3, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [HEX.teal, HEX.amber, HEX.blue], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.00", dataLabelFontSize: 9, dataLabelPosition: "outEnd", valAxisMaxVal: 1.1, valAxisMinVal: 0, valAxisLabelFormatCode: "0.0", ...chartBase("Held-out test performance (income > 50K)") });
  stat(s, 7.1, 1.25, 2.4, dsr(v, "Neural-CAR", "n_rules") + " vs " + dsr(v, "Apriori-CAR", "n_rules"), "rules in the neural vs Apriori list, same predictions");
  s.addText("Rule lists are as accurate as a tree and every prediction can be read as a sentence.", { x: 7.1, y: 2.6, w: 2.4, h: 1.8, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ============================================================ 15 fairness
{
  const s = content("9b. Ethics: rules mirror (and can proxy) bias",
    `The data are unequal: ${pct(N.gap.sex.Male)} of men vs ${pct(N.gap.sex.Female)} of women earn >50K. ${pct(N.audit.share_either)} of mined rules mention sex or race. Every classifier is below the 0.8 disparate-impact rule of thumb for sex. Removing sex and race changes DIR from 0.41 to 0.42 only, because proxies stay (relationship=Husband => sex=Male above 99.9%). Advantage of rules: the offending condition is visible and can be deleted by a reviewer.`);
  const rows = ["Apriori-CAR", "Neural-CAR", "Decision tree (depth 5)"];
  s.addChart(pres.charts.BAR, [
    { name: "With sex / race", labels: ["Apriori rules", "Neural rules", "Decision tree"], values: rows.map((m) => dsr("with sex/race", m, "sex_DIR")) },
    { name: "Without sex / race", labels: ["Apriori rules", "Neural rules", "Decision tree"], values: rows.map((m) => dsr("without sex/race", m, "sex_DIR")) }],
    { x: 0.5, y: 1.15, w: 5.2, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [HEX.red, HEX.grey], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", valAxisMaxVal: 1.0, valAxisMinVal: 0, valAxisLabelFormatCode: "0.0", ...chartBase("Disparate-impact ratio by sex (1 = parity, <0.8 = red flag)") });
  stat(s, 6.0, 1.2, 3.5, pct(N.gap.sex.Male, 0) + " vs " + pct(N.gap.sex.Female, 0), "men vs women with income >50K in the data itself");
  stat(s, 6.0, 2.3, 3.5, pct(N.audit.share_either, 0), "of mined rules mention sex or race", HEX.blue);
  s.addText(bullets(["Dropping sex/race is not enough: proxies remain", "Never use protected or proxy rules for individual decisions", "Report base-rate gaps beside every group rule"], { fontSize: 12 }), { x: 6.0, y: 3.45, w: 3.5, h: 1.6, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ============================================================ 16 privacy
{
  const s = content("9c. Privacy: k-anonymous rules, noisy verification",
    "Every rule is supported by at least 1,713 people, far above typical k-anonymity thresholds (5-20). We also tested a differentially private release: Laplace noise on the three counts used to verify each proposal (epsilon/3 each). At epsilon=1 per statistic F1 is 0.98, essentially unchanged; it breaks down only for epsilon=0.01. Caveats: epsilon is per statistic, and the network itself was not trained with DP-SGD, so only the released numbers are protected.");
  const lab = N.pr.map((r) => (r.epsilon === "inf (no noise)" ? "no noise" : String(r.epsilon)));
  s.addChart(pres.charts.LINE, [{ name: "F1 vs exact rules", labels: lab, values: N.pr.map((r) => r.F1) }, { name: "Recall", labels: lab, values: N.pr.map((r) => r.recall) }, { name: "Precision", labels: lab, values: N.pr.map((r) => r.precision) }],
    { x: 0.5, y: 1.15, w: 5.6, h: 3.9, chartColors: [HEX.teal, HEX.blue, HEX.amber], lineSize: 2.5, lineDataSymbolSize: 7, showLegend: true, legendPos: "b", showValue: false, valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.0", catAxisTitle: "epsilon per statistic", showCatAxisTitle: true, catAxisTitleFontSize: 11, ...chartBase("Utility under Laplace noise") });
  stat(s, 6.5, 1.2, 3.0, N.kanon.min_rule_support_count.toLocaleString(), "people behind the least-supported rule (min support count)");
  stat(s, 6.5, 2.4, 3.0, N.pr.find((r) => String(r.epsilon) === "1.0").F1.toFixed(2), "F1 at epsilon = 1 per statistic", HEX.blue);
  s.addText("Network training is not DP: only released statistics are protected.", { x: 6.5, y: 3.7, w: 3.0, h: 1.0, fontSize: 13, italic: true, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ============================================================ 17 limitations
{
  const s = content("10. Limitations and future improvements",
    "Be upfront: on this small dataset the neural miner is not better than Apriori. Value is the verified, explainable pipeline and the measurement of its weaknesses. The user-study protocol (20 readers, matched pairs of raw rule vs sentence vs sentence + driver, measure comprehension, time and trust) is described in the report.");
  card(s, 0.5, 1.2, 4.35, 3.8, "Limitations", "One small tabular dataset (55 items): no speed benefit, few accuracy differences\nSmall-sample gain within noise\nVerification inherits Apriori's thresholds; rules limited to 3 conditions, 1 consequent\nUnderstandability via proxies, no user study\nAssociation is not causation; 1994 data\nNetwork not trained with DP", { bs: 14 });
  card(s, 5.05, 1.2, 4.4, 3.8, "Future improvements", "High-dimensional transactions (thousands of items) where Apriori explodes\nStability selection (keep rules found in >=90% of resamples)\nMulti-item consequents, longer antecedents\nFDR-controlled thresholds, causal checks\nFairness-constrained rule selection; DP-SGD\nUser study with ~20 readers", { bs: 14, fill: C.background2 });
}

// ============================================================ 18 conclusion (dark)
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("Conclusion", { placeholder: "title" });
  s.addText(bullets([
    "Neural ARM recovers " + pct(E1["Neural (verified)"].recall_vs_apriori) + " of Apriori's " + E1["Apriori"].n_rules + " rules with exact statistics and 100% held-out replication",
    "As stable as classical mining (Jaccard " + stab("full train", "Neural (verified)").toFixed(2) + "); no significant edge against spurious rules",
    "Pruning (-59%) and plain-English rule cards make rules readable; occlusion shows which condition matters",
    "Rules mirror data bias and can proxy protected attributes: audit before use",
    "Recipe: propose with the network, verify on data, prune, explain, audit",
  ], { fontSize: 16, color: "FFFFFF" }), { x: 0.6, y: 1.25, w: 8.8, h: 3.6, margin: 0, isTextBox: true, valign: "top", color: C.background1 });
  s.addNotes("Close with the honest headline: as accurate and stable as exhaustive mining, not better, but verified, explainable and audited.");
}

// ============================================================ 19 reproducibility
{
  const s = content("11. Reproducible code and data",
    "Everything is reproducible with one command. Fixed seed 42; run takes about 4 minutes on 2 CPU cores. The report and these slides are generated from the result files so numbers always match.");
  card(s, 0.5, 1.2, 5.3, 3.8, "Project folder", "data/adult-all.csv: UCI Adult, 48,842 rows\nsrc/data.py, mining.py, neural.py, explain.py\nrun_all.py: all experiments -> results/*.csv, figures/*.png\nbuild_report.py, build_slides.js: report and slides from results\nrequirements.txt, README.md", { bs: 16 });
  card(s, 6.0, 1.2, 3.45, 3.8, "Run it", "pip install -r requirements.txt\n\npython run_all.py\n\nPython 3.13, NumPy, pandas, scikit-learn, mlxtend, statsmodels. Autoencoder written in NumPy (no GPU needed).", { bs: 15 });
}

// ============================================================ 20 thanks
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("", { placeholder: "title" });
  s.addText("Thank you", { x: 0.6, y: 1.7, w: 8.8, h: 1.0, fontSize: 44, bold: true, fontFace: "Cambria", color: C.background1, margin: 0, isTextBox: true });
  s.addText("Questions?", { x: 0.6, y: 2.8, w: 8.8, h: 0.7, fontSize: 24, color: C.accent2, margin: 0, isTextBox: true });
  s.addNotes("Q&A prep: see QA_prep.md in the project folder.");
}

(async () => {
  fs.mkdirSync("report", { recursive: true });
  await pres.writeFile({ fileName: "report/Neural_ARM_Presentation.pptx" });
  if (applyTheme) await applyTheme("report/Neural_ARM_Presentation.pptx", THEME);
  console.log("saved; theme applied:", !!applyTheme);
})();
