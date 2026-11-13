"""Neural association-rule mining (ARM-AE style) in pure NumPy.

A denoising autoencoder (DAE) is trained on the boolean transaction matrix. Because the input is randomly
masked, the network learns to *complete* partial transactions, i.e. its sigmoid output for item c given an input
that contains only itemset A approximates P(c | A).  Rule candidates A => c are the (A, c) pairs whose
reconstruction probability exceeds a threshold; every candidate is then verified on the data, so all reported
support / confidence / lift values are exact (the network only *proposes*, the data *disposes*)."""
import time
from itertools import combinations
import numpy as np
import pandas as pd

from data import attr_of
from mining import rule_key


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class DenoisingAE:
    """D -> h1 -> h2(bottleneck) -> h1 -> D with ReLU hidden layers, sigmoid output, BCE loss, Adam."""

    def __init__(self, d, h1=64, h2=16, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        sizes = [d, h1, h2, h1, d]
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(4)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(4)]
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.history = []

    def forward(self, x, keep=False):
        acts = [x]
        h = x
        for i in range(4):
            z = h @ self.W[i] + self.b[i]
            h = _sigmoid(z) if i == 3 else np.maximum(z, 0)
            acts.append(h)
        return (h, acts) if keep else h

    def encode(self, x):
        h = x
        for i in range(2):
            h = np.maximum(h @ self.W[i] + self.b[i], 0)
        return h

    def _step(self, x_in, x_tgt, lr, l2):
        out, acts = self.forward(x_in, keep=True)
        n = len(x_in)
        eps = 1e-7
        loss = -np.mean(np.sum(x_tgt * np.log(out + eps) + (1 - x_tgt) * np.log(1 - out + eps), axis=1))
        delta = (out - x_tgt) / n                      # d loss / d z_out for sigmoid + BCE
        gW, gb = [None] * 4, [None] * 4
        for i in range(3, -1, -1):
            gW[i] = acts[i].T @ delta + l2 * self.W[i]
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        params, grads = self.W + self.b, gW + gb
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def fit(self, X, epochs=30, batch=256, lr=2e-3, l2=1e-5, mask_range=(0.3, 0.95), val=None):
        """Train as a denoising AE: each row keeps a random fraction (1-mask) of its items (mask ~ U(mask_range))
        so that the network learns P(item | partial transaction); the target is the full transaction."""
        X = X.astype(np.float32)
        n = len(X)
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                xb = X[idx[s:s + batch]]
                p = self.rng.uniform(*mask_range, size=(len(xb), 1))
                x_in = xb * (self.rng.random(xb.shape) > p)
                tot += self._step(x_in, xb, lr, l2) * len(xb)
            rec = {"epoch": ep + 1, "train_loss": tot / n}
            if val is not None:
                rec["val_loss"] = self.eval_loss(val, mask_range)
            self.history.append(rec)
        return self

    def eval_loss(self, X, mask_range=(0.3, 0.95)):
        rng = np.random.default_rng(123)
        X = X.astype(np.float32)
        p = rng.uniform(*mask_range, size=(len(X), 1))
        x_in = X * (rng.random(X.shape) > p)
        out = self.forward(x_in)
        eps = 1e-7
        return float(-np.mean(np.sum(X * np.log(out + eps) + (1 - X) * np.log(1 - out + eps), axis=1)))


def frequent_antecedents(X, min_sup=0.05, max_size=3):
    """Cheap level-wise counting of antecedent itemsets (size 1..max_size, all items from distinct attributes).
    Only antecedent *support* is counted here; consequents are chosen by the network."""
    M = X.values
    n, d = M.shape
    items = list(X.columns)
    attr = np.array([attr_of(i) for i in items])
    minc = min_sup * n
    Mf = M.astype(np.float32)
    ante = []
    c1 = M.sum(0)
    L1 = [i for i in range(d) if c1[i] >= minc]
    ante += [(i,) for i in L1]
    if max_size >= 2:
        C = Mf.T @ Mf
        L2 = [(i, j) for i, j in combinations(L1, 2) if attr[i] != attr[j] and C[i, j] >= minc]
        ante += L2
    if max_size >= 3:
        L2s = set(L2)
        for (i, j) in L2:
            m = M[:, i] & M[:, j]
            cnt = m.astype(np.float32) @ Mf
            for k in L1:
                if k > j and attr[k] != attr[i] and attr[k] != attr[j] and cnt[k] >= minc \
                        and (i, k) in L2s and (j, k) in L2s:
                    ante.append((i, j, k))
    return ante


def neural_rules(ae, X, min_sup=0.05, min_conf=0.8, min_lift=1.2, tau=0.75, max_ante=3, ante=None,
                 slack=0.05):
    """Extract rules with a trained autoencoder.
    Returns (verified_rules, candidates) where candidates holds every (A, c) proposed by the network
    together with its predicted probability and (if verify) the exact statistics."""
    t0 = time.time()
    M = X.values
    n, d = M.shape
    items = list(X.columns)
    attr = np.array([attr_of(i) for i in items])
    if ante is None:
        ante = frequent_antecedents(X, min_sup, max_ante)
    V = np.zeros((len(ante), d), np.float32)
    for r, a in enumerate(ante):
        V[r, list(a)] = 1
    P = ae.forward(V)                                    # predicted P(item | antecedent)
    base = M.mean(0)
    asup = np.array([np.logical_and.reduce(M[:, list(a)], axis=1).mean() for a in ante])
    rows = []
    for r, a in enumerate(ante):
        ok = np.ones(d, bool)
        for i in a:
            ok &= attr != attr[i]                        # consequent must come from a different attribute
        # neural proposal: P(c|A) >= tau AND neural estimates of support (supp(A)*P) and lift (P/supp(c)) near their thresholds
        ok &= (P[r] >= tau) & (asup[r] * P[r] >= min_sup * (1 - slack)) & (P[r] / np.maximum(base, 1e-9) >= min_lift * (1 - slack))
        for c in np.where(ok)[0]:
            rows.append((a, c, P[r, c]))
    cand = pd.DataFrame({"a_idx": [x[0] for x in rows], "c_idx": [x[1] for x in rows], "p_nn": [x[2] for x in rows]})
    if len(cand) == 0:
        empty = pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift", "count", "key", "p_nn"])
        return empty, cand
    # exact verification on the data
    cache = {}
    sup, conf, lift, cnt = [], [], [], []
    for a, c in zip(cand["a_idx"], cand["c_idx"]):
        if a not in cache:
            m = np.ones(n, bool)
            for i in a:
                m &= M[:, i]
            cache[a] = m
        m = cache[a]
        na = m.sum()
        nac = (m & M[:, c]).sum()
        sup.append(nac / n)
        conf.append(nac / na)
        lift.append((nac / na) / base[c])
        cnt.append(nac)
    cand["support"], cand["confidence"], cand["lift"], cand["count"] = sup, conf, lift, cnt
    cand["antecedent"] = [tuple(sorted(items[i] for i in a)) for a in cand["a_idx"]]
    cand["consequent"] = [items[c] for c in cand["c_idx"]]
    cand["key"] = [rule_key(a, c) for a, c in zip(cand["antecedent"], cand["consequent"])]
    ok = (cand["support"] >= min_sup) & (cand["confidence"] >= min_conf) & (cand["lift"] >= min_lift)
    rules = cand[ok].drop(columns=["a_idx", "c_idx"]).reset_index(drop=True)
    cand["verified"] = ok.values
    rules.attrs["seconds"] = time.time() - t0
    return rules, cand


def train_ae(X, seed=0, epochs=30, h1=64, h2=16, val=None):
    ae = DenoisingAE(X.shape[1], h1, h2, seed)
    ae.fit(X.values, epochs=epochs, val=None if val is None else val.values)
    return ae


def occlusion_attribution(ae, X, antecedent, consequent):
    """Explain a rule with the network: how much does the predicted probability of the consequent drop when each
    antecedent item is removed? (leave-one-out occlusion on the reconstruction)."""
    col = {c: i for i, c in enumerate(X.columns)}
    full = np.zeros((1, X.shape[1]), np.float32)
    for it in antecedent:
        full[0, col[it]] = 1
    p_full = float(ae.forward(full)[0, col[consequent]])
    out = {}
    for it in antecedent:
        v = full.copy()
        v[0, col[it]] = 0
        out[it] = p_full - float(ae.forward(v)[0, col[consequent]])
    return p_full, out
