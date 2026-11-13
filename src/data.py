"""Data loading and transaction encoding for the UCI Adult (Census Income) dataset."""
import numpy as np
import pandas as pd

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]

SENSITIVE_ATTRS = ["sex", "race"]          # protected attributes audited in the fairness analysis
TARGET = "income"


def load_raw(path="data/adult-all.csv"):
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True, na_values="?")
    df["income"] = df["income"].str.strip().str.rstrip(".")
    return df


def discretise(df):
    """Turn every column into a categorical attribute (one 'attribute=value' item per cell).
    Missing values simply produce no item (a transaction may have fewer items)."""
    d = pd.DataFrame(index=df.index)
    d["age"] = pd.cut(df["age"], [0, 25, 35, 45, 55, 65, 120],
                      labels=["<=25", "26-35", "36-45", "46-55", "56-65", ">65"])
    d["workclass"] = df["workclass"].replace({"Self-emp-not-inc": "Self-employed", "Self-emp-inc": "Self-employed",
                                              "Local-gov": "Gov", "State-gov": "Gov", "Federal-gov": "Gov"})
    edu = df["education"].replace({"Preschool": "No-HS", "1st-4th": "No-HS", "5th-6th": "No-HS", "7th-8th": "No-HS",
                                   "9th": "No-HS", "10th": "No-HS", "11th": "No-HS", "12th": "No-HS",
                                   "Assoc-voc": "Associate", "Assoc-acdm": "Associate",
                                   "Prof-school": "Masters+", "Masters": "Masters+", "Doctorate": "Masters+"})
    d["education"] = edu
    d["marital-status"] = df["marital-status"].replace({"Married-civ-spouse": "Married", "Married-AF-spouse": "Married",
                                                        "Married-spouse-absent": "Married-absent",
                                                        "Divorced": "Divorced/Separated", "Separated": "Divorced/Separated"})
    d["occupation"] = df["occupation"]
    d["relationship"] = df["relationship"]
    d["race"] = df["race"].where(df["race"].isin(["White", "Black"]), "Other")
    d["sex"] = df["sex"]
    d["capital-gain"] = pd.cut(df["capital-gain"], [-1, 0, 5000, 1e6], labels=["none", "low", "high"])
    d["capital-loss"] = pd.cut(df["capital-loss"], [-1, 0, 1e6], labels=["none", "some"])
    d["hours-per-week"] = pd.cut(df["hours-per-week"], [-1, 34, 45, 200], labels=["part-time", "full-time", "overtime"])
    d["native-country"] = df["native-country"].map(lambda v: np.nan if pd.isna(v) else ("US" if v == "United-States" else "Non-US"))
    d["income"] = df["income"]
    return d.astype("object")


def to_onehot(cat, min_item_support=0.01):
    """Boolean transaction matrix (rows = people, columns = 'attr=value' items).
    Items rarer than `min_item_support` are dropped (they can never reach the mining threshold)."""
    cols = {}
    for a in cat.columns:
        for v in sorted(cat[a].dropna().unique()):
            cols[f"{a}={v}"] = (cat[a] == v).values
    X = pd.DataFrame(cols, index=cat.index)
    keep = X.columns[X.mean() >= min_item_support]
    return X[keep].astype(bool)


def attr_of(item):
    return item.split("=")[0]


def load_transactions(path="data/adult-all.csv"):
    raw = load_raw(path)
    cat = discretise(raw)
    X = to_onehot(cat)
    return raw, cat, X
