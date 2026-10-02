"""
core.py - data wrangling, feature engineering, ML and skill-gap logic
for the Career Skills Gap Analysis & Job Role Recommendation System.

This module has NO Streamlit / Plotly dependency, so every function can be
unit-tested or reused in a notebook.

Dataset schema expected (ai_job_market.csv):
    job_id, company_name, industry, job_title, skills_required,
    experience_level, employment_type, location, salary_range_usd,
    posted_date, company_size, tools_preferred
"""
from __future__ import annotations

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TARGET = "job_title"
SKILL_COL = "skills_required"
TOOL_COL = "tools_preferred"
EXP_COL = "experience_level"
EXP_ORDER = ["Entry", "Mid", "Senior"]
REQUIRED_COLS = [TARGET, SKILL_COL, EXP_COL]
NUM_FEATURES = ["skill_count", "tool_count"]
MODEL_COLS = ["skills", "tools", EXP_COL, *NUM_FEATURES]
CATEGORICAL_OPTIONAL = ["industry", "employment_type", "company_size", "location"]


# --------------------------------------------------------------------------- #
# 1. Loading & cleaning
# --------------------------------------------------------------------------- #
def _split_list(value) -> list[str]:
    """'Python, SQL ,python' -> ['Python', 'SQL'] (order kept, de-duplicated)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out, seen = [], set()
    for part in re.split(r"[,;|]", str(value)):
        item = part.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def _canonicalize(series_of_lists: pd.Series) -> pd.Series:
    """Unify spelling variants ('pytorch' / 'PyTorch') to the most common form."""
    counts: dict[str, Counter] = {}
    for items in series_of_lists:
        for it in items:
            counts.setdefault(it.lower(), Counter())[it] += 1
    canon = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    return series_of_lists.map(lambda L: [canon[i.lower()] for i in L])


def load_raw(source) -> pd.DataFrame:
    """Read the CSV from a path or a file-like object."""
    return pd.read_csv(source)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Automated cleaning + parsing. Returns (clean_df, report).

    Steps: normalise headers -> trim strings -> drop duplicate rows/ids ->
    drop rows with no target/skills -> impute categoricals -> parse skill & tool
    lists -> parse salary range -> parse dates -> derive numeric features.
    """
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. Found: {list(df.columns)}"
        )

    rep: dict = {"rows_raw": len(df), "cols_raw": df.shape[1]}
    rep["missing_before"] = df.isna().sum()

    # trim whitespace in text columns; blank strings -> NaN
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            s = df[c].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = s.replace("", np.nan)

    # duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    if "job_id" in df.columns:
        df = df.drop_duplicates(subset="job_id", keep="first")
    rep["duplicates_removed"] = n0 - len(df)

    # rows without target or skills are unusable
    n1 = len(df)
    df = df.dropna(subset=[TARGET, SKILL_COL])
    rep["rows_dropped_missing_core"] = n1 - len(df)

    # categorical imputation
    for c in CATEGORICAL_OPTIONAL:
        if c not in df.columns:
            df[c] = "Unknown"
    df[EXP_COL] = df[EXP_COL].astype(object).map(lambda v: v.title() if isinstance(v, str) else v)
    mode_exp = df[EXP_COL].mode().iloc[0] if df[EXP_COL].notna().any() else "Mid"
    rep["experience_imputed"] = int(df[EXP_COL].isna().sum())
    df[EXP_COL] = df[EXP_COL].fillna(mode_exp)
    cat_filled = 0
    for c in CATEGORICAL_OPTIONAL:
        cat_filled += int(df[c].isna().sum())
        df[c] = df[c].fillna("Unknown")
    rep["categorical_imputed"] = cat_filled

    # list parsing (after de-duplication, because lists are unhashable)
    df["skills"] = _canonicalize(df[SKILL_COL].map(_split_list))
    tools_src = df[TOOL_COL] if TOOL_COL in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    df["tools"] = _canonicalize(tools_src.map(_split_list))
    n2 = len(df)
    df = df[df["skills"].map(len) > 0].copy()
    rep["rows_dropped_empty_skills"] = n2 - len(df)
    df["skill_count"] = df["skills"].map(len)
    df["tool_count"] = df["tools"].map(len)

    # ordinal experience (kept for plots; one-hot is used for the model)
    ord_map = {lvl: i for i, lvl in enumerate(EXP_ORDER)}
    df["exp_ord"] = df[EXP_COL].map(ord_map)
    df["exp_ord"] = df["exp_ord"].fillna(df["exp_ord"].median())

    # salary range "92860-109598" -> min / max / mid
    if "salary_range_usd" in df.columns:
        nums = (
            df["salary_range_usd"].astype(str).str.replace(",", "", regex=False)
            .str.extract(r"(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)").astype(float)
        )
        lo, hi = nums.min(axis=1), nums.max(axis=1)
    else:
        lo = hi = pd.Series(np.nan, index=df.index)
    df["salary_min"], df["salary_max"] = lo, hi
    rep["salary_unparseable"] = int(df["salary_min"].isna().sum())
    for col in ("salary_min", "salary_max"):
        grp = df.groupby([TARGET, EXP_COL])[col].transform("median")
        df[col] = df[col].fillna(grp).fillna(df[col].median())
    df["salary_mid"] = (df["salary_min"] + df["salary_max"]) / 2

    # dates
    if "posted_date" in df.columns:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
        df["posted_month"] = df["posted_date"].dt.to_period("M").dt.to_timestamp()

    df = df.reset_index(drop=True)
    rep["rows_clean"] = len(df)
    rep["missing_after"] = df.drop(columns=["skills", "tools"]).isna().sum()
    return df, rep


def skill_vocab(df: pd.DataFrame, col: str = "skills") -> list[str]:
    return sorted({s for L in df[col] for s in L})


def multi_hot(df: pd.DataFrame, col: str = "skills", prefix: str = "") -> pd.DataFrame:
    """Multi-hot (binary) matrix for a list-valued column."""
    mlb = MultiLabelBinarizer()
    m = mlb.fit_transform(df[col])
    return pd.DataFrame(m, columns=[f"{prefix}{c}" for c in mlb.classes_], index=df.index)


def skill_counts(df: pd.DataFrame, col: str = "skills") -> pd.Series:
    return pd.Series(Counter(s for L in df[col] for s in L)).sort_values(ascending=False)


def dataset_kpis(df: pd.DataFrame) -> dict:
    sc = skill_counts(df)
    mean_exp = float(df["exp_ord"].mean())
    return {
        "n_rows": len(df),
        "n_roles": df[TARGET].nunique(),
        "top_skill": sc.index[0],
        "top_skill_share": sc.iloc[0] / len(df),
        "avg_exp_label": EXP_ORDER[int(round(mean_exp))],
        "avg_exp_value": mean_exp,
        "avg_skills": float(df["skill_count"].mean()),
        "median_salary": float(df["salary_mid"].median()),
    }


# --------------------------------------------------------------------------- #
# 2. Feature engineering (sklearn-compatible so it can sit inside a Pipeline)
# --------------------------------------------------------------------------- #
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Multi-hot(skills) + multi-hot(tools) + One-Hot(experience level)
    + StandardScaler(skill_count, tool_count).

    Fitted ONLY on training folds when used inside a Pipeline -> no leakage.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.skill_mlb_ = MultiLabelBinarizer().fit(X["skills"])
        self.tool_mlb_ = MultiLabelBinarizer().fit(X["tools"])
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[[EXP_COL]])
        self.scaler_ = StandardScaler().fit(X[NUM_FEATURES])
        self.feature_names_ = (
            [f"skill_{c}" for c in self.skill_mlb_.classes_]
            + [f"tool_{c}" for c in self.tool_mlb_.classes_]
            + [f"exp_{c}" for c in self.ohe_.categories_[0]]
            + [f"scaled_{c}" for c in NUM_FEATURES]
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # unseen labels are ignored on purpose
            sk = self.skill_mlb_.transform(X["skills"])
            tl = self.tool_mlb_.transform(X["tools"])
        ex = self.ohe_.transform(X[[EXP_COL]])
        nm = self.scaler_.transform(X[NUM_FEATURES])
        return np.hstack([sk, tl, ex, nm]).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def feature_type(name: str) -> str:
    if name.startswith("skill_"):
        return "Skill"
    if name.startswith("tool_"):
        return "Preferred tool"
    if name.startswith("exp_"):
        return "Experience level"
    return "Numeric (scaled)"


def candidate_frame(skills: list[str], tools: list[str], experience: str) -> pd.DataFrame:
    """Turn the Streamlit form inputs into one model-ready row."""
    return pd.DataFrame(
        {
            "skills": [list(skills)],
            "tools": [list(tools)],
            EXP_COL: [experience],
            "skill_count": [len(skills)],
            "tool_count": [len(tools)],
        }
    )


# --------------------------------------------------------------------------- #
# 3. Model training & evaluation
# --------------------------------------------------------------------------- #
def _model_zoo(seed: int) -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        ),
        "Multinomial Logistic Regression": LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, cv_folds: int = 5) -> dict:
    X, y_raw = df[MODEL_COLS], df[TARGET]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    results, pipes = {}, {}
    for name, clf in _model_zoo(seed).items():
        pipe = Pipeline([("features", FeatureBuilder()), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        cv = cross_validate(clone(pipe), X, y, cv=skf, scoring=["accuracy", "f1_weighted"], n_jobs=1)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
        results[name] = {
            "Accuracy": accuracy_score(yte, pred),
            "Precision (weighted)": precision_score(yte, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(yte, pred, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(yte, pred, average="weighted", zero_division=0),
            "F1 (macro)": f1_score(yte, pred, average="macro", zero_division=0),
            f"CV accuracy ({cv_folds}-fold) mean": cv["test_accuracy"].mean(),
            "CV accuracy std": cv["test_accuracy"].std(),
            "_cm": confusion_matrix(yte, pred, labels=range(len(classes))),
            "_pred": pred,
        }
        pipes[name] = pipe

    # Does the best real model beat guessing the majority class? (one-sided binomial test)
    majority_share = np.bincount(ytr).max() / len(ytr)
    for name in results:
        k = int((results[name]["_pred"] == yte).sum())
        results[name]["p_vs_baseline"] = stats.binomtest(k, len(yte), majority_share, alternative="greater").pvalue

    return {
        "classes": classes,
        "pipelines": pipes,
        "results": results,
        "y_test": yte,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "chance": 1 / len(classes),
        "majority_share": majority_share,
        "cv_folds": cv_folds,
    }


def metrics_table(bundle: dict) -> pd.DataFrame:
    rows = {
        n: {k: v for k, v in r.items() if not k.startswith("_") and k != "p_vs_baseline"}
        for n, r in bundle["results"].items()
    }
    return pd.DataFrame(rows).T


def per_class_report(bundle: dict, model: str) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        bundle["y_test"], bundle["results"][model]["_pred"],
        labels=range(len(bundle["classes"])), zero_division=0,
    )
    return pd.DataFrame({"Role": bundle["classes"], "Precision": p, "Recall": r, "F1": f, "Support": s})


def feature_importance(bundle: dict, model: str) -> pd.DataFrame:
    pipe = bundle["pipelines"][model]
    names = pipe.named_steps["features"].feature_names_
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance", "type"])
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    out = pd.DataFrame({"feature": names, "importance": imp})
    out["type"] = out["feature"].map(feature_type)
    out["label"] = out["feature"].str.replace(r"^(skill_|tool_|exp_|scaled_)", "", regex=True)
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def predict_role_probs(bundle: dict, model: str, frame: pd.DataFrame) -> pd.Series:
    proba = bundle["pipelines"][model].predict_proba(frame)[0]
    return pd.Series(proba, index=bundle["classes"]).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 4. Statistical "signal audit"
# --------------------------------------------------------------------------- #
def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def signal_audit(df: pd.DataFrame) -> dict:
    """Chi-square tests: is each skill / attribute statistically linked to the job role?"""
    vocab = skill_vocab(df)
    rows = []
    for s in vocab:
        has = df["skills"].map(lambda L, s=s: s in L)
        ct = pd.crosstab(df[TARGET], has)
        if ct.shape[1] < 2:
            chi, p = 0.0, 1.0
        else:
            chi, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"skill": s, "chi2": chi, "p_value": p})
    sk = pd.DataFrame(rows)
    sk["p_adj (BH-FDR)"] = _bh_adjust(sk["p_value"].values)
    sk["significant (5%)"] = sk["p_adj (BH-FDR)"] < 0.05
    sk = sk.sort_values("p_value").reset_index(drop=True)

    other = []
    for c in [EXP_COL, "industry", "company_size", "employment_type"]:
        if c in df.columns and df[c].nunique() > 1:
            chi, p, _, _ = stats.chi2_contingency(pd.crosstab(df[TARGET], df[c]))
            other.append({"attribute": f"{c} vs role (chi-square)", "statistic": chi, "p_value": p})
    groups = [g["salary_mid"].values for _, g in df.groupby(TARGET)]
    if len(groups) > 1:
        f, p = stats.f_oneway(*groups)
        other.append({"attribute": "salary midpoint across roles (ANOVA)", "statistic": f, "p_value": p})
    other = pd.DataFrame(other)
    other["significant (5%)"] = other["p_value"] < 0.05
    return {"skills": sk, "other": other, "n_significant": int(sk["significant (5%)"].sum()), "n_skills": len(vocab)}


# --------------------------------------------------------------------------- #
# 5. Role profiles & skill-gap analysis
# --------------------------------------------------------------------------- #
def role_skill_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = roles, columns = skills, values = share of that role's postings listing the skill."""
    mh = multi_hot(df)
    mh[TARGET] = df[TARGET].values
    return mh.groupby(TARGET).mean()


def role_profile(df: pd.DataFrame, role: str, experience: str | None = None, min_n: int = 30):
    """
    Skill demand profile of a role. Optionally restrict to one experience level
    (only if >= min_n postings exist, otherwise falls back to all levels).
    Returns (share Series sorted desc, n_postings_used, experience_filter_applied).
    """
    sub = df[df[TARGET] == role]
    applied = False
    if experience:
        sub_e = sub[sub[EXP_COL] == experience]
        if len(sub_e) >= min_n:
            sub, applied = sub_e, True
    vocab = skill_vocab(df)
    share = sub["skills"].explode().value_counts().reindex(vocab).fillna(0) / max(len(sub), 1)
    return share.sort_values(ascending=False), len(sub), applied


def analyze_gap(candidate_skills: list[str], profile: pd.Series, overall_share: pd.Series, top_k: int = 8) -> dict:
    cand = set(candidate_skills)
    req = profile.head(top_k)
    rows = []
    for rank, (skill, share) in enumerate(req.items(), start=1):
        priority = "Critical" if rank <= 3 else ("Important" if rank <= 6 else "Nice to have")
        rows.append(
            {
                "rank": rank,
                "skill": skill,
                "demand_pct": share * 100,
                "lift": share / overall_share[skill] if overall_share[skill] > 0 else np.nan,
                "priority": priority,
                "have": skill in cand,
            }
        )
    table = pd.DataFrame(rows)
    readiness = float(table.loc[table["have"], "demand_pct"].sum() / table["demand_pct"].sum()) if len(table) else 0.0
    return {
        "table": table,
        "readiness": readiness,
        "have": table[table["have"]],
        "missing": table[~table["have"]].reset_index(drop=True),
        "extra": sorted(cand - set(req.index)),
    }


def match_all_roles(candidate_skills: list[str], matrix: pd.DataFrame, top_k: int = 8) -> pd.Series:
    cand, out = set(candidate_skills), {}
    for role, row in matrix.iterrows():
        top = row.nlargest(top_k)
        out[role] = float(top[top.index.isin(cand)].sum() / top.sum()) if top.sum() else 0.0
    return pd.Series(out).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 6. Learning roadmap (curated knowledge - NOT derived from the dataset)
# --------------------------------------------------------------------------- #
PHASES = ["Phase 1 - Foundations", "Phase 2 - Core ML & analytics", "Phase 3 - Specialisation", "Phase 4 - Deployment & cloud"]

SKILL_META: dict[str, dict] = {
    "Python": dict(phase=0, weeks=4, learn="Official Python tutorial + a small automation project", cert="PCAP - Certified Associate in Python Programming"),
    "R": dict(phase=0, weeks=3, learn="'R for Data Science' (free book)", cert="-"),
    "SQL": dict(phase=0, weeks=3, learn="Practice joins, window functions and CTEs on real tables", cert="HackerRank SQL certificate"),
    "Excel": dict(phase=0, weeks=2, learn="Pivot tables, lookups, Power Query", cert="Microsoft Office Specialist: Excel"),
    "NumPy": dict(phase=0, weeks=1, learn="Official NumPy quickstart", cert="-"),
    "Pandas": dict(phase=0, weeks=2, learn="Kaggle Learn: Pandas, then clean a messy CSV", cert="Kaggle Learn certificate"),
    "C++": dict(phase=0, weeks=6, learn="Modern C++ basics, STL, memory model", cert="C++ Institute CPA"),
    "Scikit-learn": dict(phase=1, weeks=3, learn="Scikit-learn user guide + Kaggle Intermediate ML", cert="Kaggle Learn certificate"),
    "Power BI": dict(phase=1, weeks=3, learn="Build a dashboard with DAX measures", cert="Microsoft PL-300 Power BI Data Analyst"),
    "Keras": dict(phase=1, weeks=2, learn="Keras guides: train a small CNN / MLP", cert="-"),
    "TensorFlow": dict(phase=1, weeks=4, learn="DeepLearning.AI TensorFlow Developer course", cert="DeepLearning.AI TensorFlow Developer (Coursera)"),
    "PyTorch": dict(phase=1, weeks=4, learn="Official PyTorch tutorials or fast.ai", cert="-"),
    "Reinforcement Learning": dict(phase=2, weeks=6, learn="Sutton & Barto + Hugging Face Deep RL course", cert="-"),
    "Hugging Face": dict(phase=2, weeks=3, learn="Hugging Face LLM / NLP course (free)", cert="-"),
    "LangChain": dict(phase=2, weeks=2, learn="LangChain Academy courses; build a RAG demo", cert="-"),
    "CUDA": dict(phase=2, weeks=5, learn="NVIDIA DLI: Fundamentals of Accelerated Computing", cert="NVIDIA DLI certificate"),
    "FastAPI": dict(phase=3, weeks=2, learn="Official FastAPI tutorial; serve a model endpoint", cert="-"),
    "Flask": dict(phase=3, weeks=2, learn="Official Flask quickstart; build a small API", cert="-"),
    "MLflow": dict(phase=3, weeks=2, learn="MLflow tracking & model registry tutorials", cert="-"),
    "AWS": dict(phase=3, weeks=5, learn="Core services (S3, EC2, IAM, SageMaker)", cert="AWS Certified Machine Learning Engineer - Associate"),
    "Azure": dict(phase=3, weeks=5, learn="Azure ML + storage + identity basics", cert="Microsoft Azure Data Scientist Associate (DP-100)"),
    "GCP": dict(phase=3, weeks=5, learn="BigQuery + Vertex AI basics", cert="Google Cloud Professional Machine Learning Engineer"),
}

ROLE_PROJECTS = {
    "Data Analyst": "Build an interactive dashboard that answers a real business question (SQL + BI tool) and present the insights.",
    "Data Scientist": "Deliver an end-to-end predictive project (cleaning -> model -> evaluation -> short report) on GitHub.",
    "ML Engineer": "Train a model and ship it as a versioned API with experiment tracking and automated deployment.",
    "NLP Engineer": "Fine-tune and evaluate a transformer for a text task, then wrap it in a small LLM-powered app.",
    "Computer Vision Engineer": "Train an image classifier/detector and benchmark GPU inference speed and accuracy.",
    "AI Researcher": "Reproduce a recent paper's results and write up your own ablation study.",
    "Quant Researcher": "Backtest a systematic strategy with strict out-of-sample validation and risk metrics.",
    "AI Product Manager": "Write a PRD with success metrics for an ML feature and prototype it with a low-code LLM demo.",
}
_DEFAULT_META = dict(phase=1, weeks=2, learn="Official documentation + one small hands-on project", cert="-")


def build_roadmap(missing: pd.DataFrame, role: str) -> dict:
    """Group missing skills into ordered learning phases (priority first inside a phase)."""
    prio_rank = {"Critical": 0, "Important": 1, "Nice to have": 2}
    steps = []
    for _, r in missing.iterrows():
        meta = {**_DEFAULT_META, **SKILL_META.get(r["skill"], {})}
        steps.append({**meta, "skill": r["skill"], "priority": r["priority"], "demand_pct": r["demand_pct"]})
    steps.sort(key=lambda s: (s["phase"], prio_rank[s["priority"]], -s["demand_pct"]))
    phases = []
    for i, name in enumerate(PHASES):
        items = [s for s in steps if s["phase"] == i]
        if items:
            phases.append({"name": name, "steps": items, "weeks": sum(s["weeks"] for s in items)})
    return {
        "phases": phases,
        "total_weeks": sum(p["weeks"] for p in phases),
        "project": ROLE_PROJECTS.get(role, "Build one portfolio project that uses every new skill together."),
    }


def roadmap_markdown(role: str, readiness: float, roadmap: dict) -> str:
    lines = [f"# Learning roadmap for: {role}", f"Current skill match: {readiness * 100:.0f}%",
             f"Estimated effort: ~{roadmap['total_weeks']} weeks (rough guide)", ""]
    if not roadmap["phases"]:
        lines.append("No gaps found among the required skills - focus on portfolio projects.")
    for ph in roadmap["phases"]:
        lines.append(f"## {ph['name']} (~{ph['weeks']} weeks)")
        for s in ph["steps"]:
            cert = f" | Cert: {s['cert']}" if s["cert"] != "-" else ""
            lines.append(f"- [{s['priority']}] {s['skill']} - {s['learn']}{cert}")
        lines.append("")
    lines.append(f"## Capstone project\n{roadmap['project']}")
    return "\n".join(lines)
