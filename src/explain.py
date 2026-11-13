"""Human-readable explanations for rules: natural-language templates + evidence + neural occlusion attribution."""
import numpy as np
from data import attr_of

PHRASE = {
    "age": lambda v: f"are aged {v}" if not v.startswith(("<=", ">")) else (f"are {v[2:]} or younger" if v.startswith("<=") else f"are over {v[1:]}"),
    "workclass": lambda v: f"work in the {v} sector",
    "education": lambda v: f"have {v} level education" if v not in ("HS-grad", "Some-college") else ("finished high school (no college)" if v == "HS-grad" else "attended some college"),
    "marital-status": lambda v: f"are {v.lower()}" if v != "Divorced/Separated" else "are divorced or separated",
    "occupation": lambda v: f"work as {v}",
    "relationship": lambda v: f"have the household role '{v}'",
    "race": lambda v: f"have race recorded as {v}",
    "sex": lambda v: f"are {v.lower()}",
    "capital-gain": lambda v: {"none": "have no capital gains", "low": "have small capital gains (<$5k)", "high": "have large capital gains (>$5k)"}[v],
    "capital-loss": lambda v: {"none": "have no capital losses", "some": "have some capital losses"}[v],
    "hours-per-week": lambda v: {"part-time": "work part-time (<35 h/week)", "full-time": "work full-time (35-45 h/week)", "overtime": "work overtime (>45 h/week)"}[v],
    "native-country": lambda v: "were born in the US" if v == "US" else "were born outside the US",
    "income": lambda v: f"earn {'more than $50K' if v == '>50K' else '$50K or less'} per year",
}
PHRASE_C = {  # consequent phrasing: 'tend to ...'
    "income": lambda v: "earn more than $50K per year" if v == ">50K" else "earn $50K or less per year",
}


def item_text(item):
    a, v = item.split("=", 1)
    return PHRASE[a](v)


def explain_rule(row, base_rate=None, drivers=None, n_total=None):
    ante = list(row["antecedent"])
    cons = row["consequent"]
    cond = ", ".join(item_text(i) for i in ante[:-1]) + (" and " if len(ante) > 1 else "") + item_text(ante[-1])
    conf = row["confidence"]
    n_match = int(round(row["count"] / conf))
    s = (f"People who {cond} {item_text(cons)} in {conf:.0%} of cases "
         f"({int(row['count']):,} of {n_match:,} matching people; {row['lift']:.1f}x the base rate).")
    if drivers and len(ante) > 1:
        top = max(drivers, key=drivers.get)
        s += f" Strongest driver according to the network: '{top}'."
    return s
