"""Data prep and statistics for The Dean's Dilemma app."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data" / "Synthetic_Dean_Dilemma_65_35.xlsx"
GROUP_ORDER = ["75+", "50–75", "Below 50", "No test", "Missing score"]


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA)
    df["Placed"] = (df["Placement"] == "Placed").astype(int)
    df["Test_Taken"] = df["Entrance_Test"].notna()
    # Zero percentile among test-takers is treated as missing, not a zero score
    df["ET_Score"] = df["Percentile_ET"].where(df["Test_Taken"] & (df["Percentile_ET"] > 0))

    def grp(r):
        if not r["Test_Taken"]:
            return "No test"
        if pd.isna(r["ET_Score"]):
            return "Missing score"
        if r["ET_Score"] >= 75:
            return "75+"
        if r["ET_Score"] >= 50:
            return "50–75"
        return "Below 50"

    df["Entrance_Group"] = df.apply(grp, axis=1)
    df["Entrance_Group"] = pd.Categorical(df["Entrance_Group"], GROUP_ORDER, ordered=True)
    df["Gender_Label"] = df["Gender"].map({"M": "Male", "F": "Female"})
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def placement_by_group(df, col):
    g = df.groupby(col, observed=True)["Placed"].agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    ci = g.apply(lambda r: wilson(r["sum"], r["count"]), axis=1)
    g["lo"] = [c[0] for c in ci]
    g["hi"] = [c[1] for c in ci]
    return g


def rq1_odds_ratios(df):
    """Adjusted ORs for entrance group vs 75+ (controls: SSC, degree %, experience)."""
    X = pd.get_dummies(df["Entrance_Group"], dtype=float).drop(columns="75+")
    for c in ["Percent_SSC", "Percent_Degree", "Experience_Yrs"]:
        X[c] = (df[c] - df[c].mean()) / df[c].std()
    m = sm.Logit(df["Placed"], sm.add_constant(X)).fit(disp=0)
    out = pd.DataFrame({"OR": np.exp(m.params), "lo": np.exp(m.conf_int()[0]),
                        "hi": np.exp(m.conf_int()[1]), "p": m.pvalues})
    return out.drop(index="const")


def rq2_models(df):
    cols = ["Percent_SSC", "Percent_HSC", "Percent_Degree", "Experience_Yrs", "ET_Score"]
    d = df.copy()
    d["ET_Missing"] = d["ET_Score"].isna().astype(float)
    d["ET_Score"] = d["ET_Score"].fillna(d["ET_Score"].mean())
    X = d[cols].copy()
    X = (X - X.mean()) / X.std()
    X["ET_Missing"] = d["ET_Missing"]
    lg = sm.Logit(d["Placed"], sm.add_constant(X)).fit(disp=0)
    odds = pd.DataFrame({"OR": np.exp(lg.params), "p": lg.pvalues}).drop(index="const")
    placed = d["Placed"] == 1
    ols = sm.OLS(np.log(d.loc[placed, "Salary"]), sm.add_constant(X[placed])).fit()
    sal = pd.DataFrame({"pct_per_SD": (np.exp(ols.params) - 1) * 100, "p": ols.pvalues}).drop(index="const")
    vif = pd.Series({c: 1 / (1 - sm.OLS(X[c], sm.add_constant(X.drop(columns=c))).fit().rsquared)
                     for c in cols})
    return odds, sal, ols.rsquared, vif


def cutoff_curve(df, col, grid):
    base = df if col != "ET_Score" else df[df["ET_Score"].notna()]
    rows = []
    for t in grid:
        k = base[base[col] >= t]
        rows.append({"cutoff": t, "kept": len(k) / len(base),
                     "placement": k["Placed"].mean() if len(k) else np.nan})
    return pd.DataFrame(rows), base["Placed"].mean()


def gender_stats(df, n_boot=2000, seed=0):
    m = df[(df.Gender == "M") & (df.Placed == 1)]["Salary"].values
    f = df[(df.Gender == "F") & (df.Placed == 1)]["Salary"].values
    gap = f.mean() - m.mean()
    _, p_t = stats.ttest_ind(f, m, equal_var=False)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(f, len(f)).mean() - rng.choice(m, len(m)).mean() for _ in range(n_boot)]
    ct = pd.crosstab(df["Gender"], df["Placed"])
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    return {"gap_F_minus_M": gap, "p_salary": p_t, "ci": np.percentile(boots, [2.5, 97.5]),
            "chi2": chi2, "p_place": p_chi}


FEATURES = ["Percent_SSC", "Percent_HSC", "Percent_Degree", "Experience_Yrs",
            "ET_Score", "ET_Missing", "Test_Taken_I", "Percent_MBA"]


def model_frame(df, include_mba=False):
    d = pd.DataFrame(index=df.index)
    for c in ["Percent_SSC", "Percent_HSC", "Percent_Degree", "Experience_Yrs"]:
        d[c] = df[c]
    d["ET_Missing"] = df["ET_Score"].isna().astype(int)
    d["ET_Score"] = df["ET_Score"].fillna(df["ET_Score"].median())
    d["Test_Taken_I"] = df["Test_Taken"].astype(int)
    if include_mba:
        d["Percent_MBA"] = df["Percent_MBA"]
    return d


def make_models():
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Bagging": BaggingClassifier(n_estimators=100, random_state=0, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    }


def cv_models(df, n_splits=5, n_repeats=2):
    """Repeated CV; target = NOT placed (the at-risk class)."""
    X = model_frame(df)
    y = 1 - df["Placed"]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    rows = []
    for name, mdl in make_models().items():
        r = cross_validate(mdl, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "recall": "recall",
                                   "precision": make_scorer(precision_score, zero_division=0)},
                           n_jobs=1)
        rows.append({"Model": name, "AUC": r["test_roc_auc"].mean(), "AUC_sd": r["test_roc_auc"].std(),
                     "Recall (Not placed)": r["test_recall"].mean(),
                     "Precision (Not placed)": r["test_precision"].mean()})
    return pd.DataFrame(rows)


def fit_risk_model(df):
    X = model_frame(df)
    y = 1 - df["Placed"]
    mdl = GradientBoostingClassifier(random_state=0).fit(X, y)
    return mdl, X.columns.tolist(), pd.Series(mdl.feature_importances_, index=X.columns).sort_values()
