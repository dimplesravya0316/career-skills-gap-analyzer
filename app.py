"""
Career Skills Gap Analysis & Job Role Recommendation System
Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Career Skills Gap Analyzer", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg,#4f46e5,#9333ea 60%,#db2777); color:#fff;
       padding: 26px 32px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin:0; font-size: 2rem; color:#fff;}
.hero p {margin:6px 0 0 0; opacity:.92; font-size:1.02rem;}
.kpi {background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(236,72,153,.10));
      border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 14px 18px; height: 100%;}
.kpi .label {font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; opacity:.75;}
.kpi .value {font-size:1.65rem; font-weight:700; line-height:1.25;}
.kpi .sub {font-size:.8rem; opacity:.7;}
.section-title {font-size:1.25rem; font-weight:700; margin: 8px 0 4px 0;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; margin:2px 4px 2px 0;
       border:1px solid rgba(128,128,128,.35);}
.pill.ok {background: rgba(34,197,94,.18);} .pill.miss {background: rgba(239,68,68,.16);}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_CSV = Path(__file__).with_name("ai_job_market.csv")
NAV = [
    "🏠 Executive Summary & Overview",
    "🧹 Data Preprocessing & Skill Mapping",
    "📊 Interactive Exploratory Data Analysis (EDA)",
    "🤖 Machine Learning Models",
    "🎯 Skills Gap & Job Recommendation",
]


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def pills(items: list[str], kind: str) -> str:
    return "".join(f'<span class="pill {kind}">{html.escape(i)}</span>' for i in items) or "—"


# --------------------------------------------------------------------------- #
# Cached data / model loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and cleaning data …")
def load_data(file_bytes: bytes):
    key = hashlib.md5(file_bytes).hexdigest()
    raw = core.load_raw(io.BytesIO(file_bytes))
    df, rep = core.clean(raw)
    return raw, df, rep, key


@st.cache_resource(show_spinner="Training models (cross-validation included) …")
def get_models(data_key: str, _df: pd.DataFrame, test_size: float, seed: int):
    return core.train_and_evaluate(_df, test_size=test_size, seed=seed)


@st.cache_data(show_spinner=False)
def get_audit(data_key: str, _df: pd.DataFrame):
    return core.signal_audit(_df)


# --------------------------------------------------------------------------- #
# Sidebar: navigation + data source
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🎯 Career Skills Gap Analyzer")
page = st.sidebar.radio("Navigate", NAV, label_visibility="collapsed")
st.sidebar.divider()
with st.sidebar.expander("📁 Data source", expanded=False):
    upload = st.file_uploader("Upload a CSV with the same schema", type="csv")
    st.caption("Default: `ai_job_market.csv` placed next to `app.py`.")

if upload is not None:
    file_bytes = upload.getvalue()
elif DEFAULT_CSV.exists():
    file_bytes = DEFAULT_CSV.read_bytes()
else:
    st.error("`ai_job_market.csv` not found next to app.py. Put it there or upload a CSV in the sidebar.")
    st.stop()

try:
    raw, df, rep, DATA_KEY = load_data(file_bytes)
except ValueError as e:
    st.error(str(e))
    st.stop()

VOCAB = core.skill_vocab(df)
TOOLS = core.skill_vocab(df, "tools")
ROLES = sorted(df[core.TARGET].unique())
MATRIX = core.role_skill_matrix(df)
OVERALL = core.skill_counts(df).reindex(MATRIX.columns).fillna(0) / len(df)
st.sidebar.caption(f"{len(df):,} postings · {len(ROLES)} roles · {len(VOCAB)} skills")


# =========================================================================== #
# PAGE 1 - Executive summary
# =========================================================================== #
def page_overview() -> None:
    st.markdown(
        '<div class="hero"><h1>Career Skills Gap Analysis & Job Role Recommendation</h1>'
        "<p>Talent analytics on the AI job market: what employers ask for, how roles differ, "
        "and what you still need to learn for your target role.</p></div>",
        unsafe_allow_html=True,
    )
    k = core.dataset_kpis(df)
    c = st.columns(5)
    kpi(c[0], "Total profiles (job postings)", f"{k['n_rows']:,}", "rows after cleaning")
    kpi(c[1], "Unique job roles", str(k["n_roles"]), "target classes")
    kpi(c[2], "Most in-demand skill", k["top_skill"], f"in {k['top_skill_share']:.0%} of postings")
    kpi(c[3], "Average experience level", k["avg_exp_label"], f"ordinal mean {k['avg_exp_value']:.2f} (0=Entry, 2=Senior)")
    kpi(c[4], "Median salary (midpoint)", f"${k['median_salary']:,.0f}", f"avg {k['avg_skills']:.1f} skills / posting")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Postings per job role</div>', unsafe_allow_html=True)
        vc = df[core.TARGET].value_counts().reset_index()
        vc.columns = ["Job role", "Postings"]
        fig = px.bar(vc, x="Postings", y="Job role", orientation="h", text="Postings", color="Postings",
                     color_continuous_scale="Purples")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=380,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
    with right:
        st.markdown('<div class="section-title">Postings over time</div>', unsafe_allow_html=True)
        if "posted_month" in df.columns and df["posted_month"].notna().any():
            ts = df.groupby("posted_month").size().reset_index(name="Postings")
            fig = px.area(ts, x="posted_month", y="Postings", labels={"posted_month": "Month"})
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig)
        else:
            st.info("No usable `posted_date` column.")

    st.markdown('<div class="section-title">How this app works</div>', unsafe_allow_html=True)
    a, b, c3 = st.columns(3)
    a.markdown("**1 · Prepare**\n\nClean the data, parse comma-separated skills into multi-hot vectors, "
               "encode categories and scale numeric inputs.")
    b.markdown("**2 · Explore & model**\n\nInteractive EDA, then Random Forest vs Multinomial Logistic Regression "
               "(plus a majority-class baseline) with full evaluation.")
    c3.markdown("**3 · Recommend**\n\nEnter your skills and target role to get a match score, a skill-gap chart, "
                "a missing-skills report and a learning roadmap.")

    audit = get_audit(DATA_KEY, df)
    st.markdown('<div class="section-title">📌 Data-quality finding you should know about</div>', unsafe_allow_html=True)
    st.warning(
        f"**This dataset is a table of job postings, not candidate profiles** — there are no candidate "
        f"years of experience, test scores, certifications, education or current salary columns. The app therefore "
        f"uses the posting's `experience_level`, `skills_required`, `tools_preferred` and `salary_range_usd`.\n\n"
        f"**Statistical audit:** only **{audit['n_significant']} of {audit['n_skills']}** skills are significantly "
        f"associated with the job title (chi-square, FDR-corrected). Skills are spread almost uniformly across roles, "
        f"so any classifier will perform close to chance. Open the *Machine Learning Models* page for the numbers. "
        f"This is a property of the data (it looks synthetic), not a bug in the code."
    )


# =========================================================================== #
# PAGE 2 - Preprocessing & skill mapping
# =========================================================================== #
def page_preprocessing() -> None:
    st.title("🧹 Data Preprocessing & Skill Mapping")
    t1, t2, t3, t4, t5 = st.tabs(
        ["1 · Raw data & quality", "2 · Cleaning & parsing", "3 · Skill mapping (multi-hot)",
         "4 · Encoding & scaling", "5 · Final feature matrix"]
    )

    with t1:
        c = st.columns(4)
        kpi(c[0], "Raw rows × cols", f"{rep['rows_raw']:,} × {rep['cols_raw']}")
        kpi(c[1], "Duplicates removed", str(rep["duplicates_removed"]))
        kpi(c[2], "Rows dropped (no role/skills)", str(rep["rows_dropped_missing_core"] + rep["rows_dropped_empty_skills"]))
        kpi(c[3], "Clean rows", f"{rep['rows_clean']:,}")
        st.markdown("**Raw data preview**")
        st.dataframe(raw.head(10), hide_index=True)
        mv = pd.DataFrame({"Missing before": rep["missing_before"]})
        mv["Missing after cleaning"] = rep["missing_after"].reindex(mv.index).fillna(0).astype(int)
        mv = mv[(mv > 0).any(axis=1)] if (mv > 0).any().any() else mv
        st.markdown("**Missing values (before → after)**")
        if (rep["missing_before"] == 0).all():
            st.success("No missing values were found in the raw file. The imputation logic is still active "
                       "for any CSV you upload.")
        st.dataframe(mv.reset_index().rename(columns={"index": "column"}), hide_index=True)
        st.markdown("**Mapping of your project spec to this dataset**")
        st.dataframe(pd.DataFrame({
            "Spec asked for": ["Target role", "Technical skills", "Experience level", "Salary", "Education / certifications / soft skills / test scores"],
            "Used from dataset": ["job_title", "skills_required (+ tools_preferred)", "experience_level (Entry / Mid / Senior)",
                                  "salary_range_usd → min / max / midpoint", "Not present in the dataset"],
        }), hide_index=True)

    with t2:
        st.markdown("**Automated steps**")
        st.markdown(
            f"- Headers normalised, whitespace trimmed, blank strings → missing\n"
            f"- Exact duplicate rows / duplicate `job_id` removed: **{rep['duplicates_removed']}**\n"
            f"- Rows without role or skills dropped: **{rep['rows_dropped_missing_core'] + rep['rows_dropped_empty_skills']}**\n"
            f"- Missing `experience_level` → mode: **{rep['experience_imputed']}**; other categoricals → `Unknown`: **{rep['categorical_imputed']}**\n"
            f"- Skill / tool strings split on `,` `;` `|`, de-duplicated, spelling variants unified\n"
            f"- Salary range parsed to min / max / midpoint (unparseable → group median): **{rep['salary_unparseable']}**\n"
            f"- Posting dates parsed to datetime"
        )
        st.markdown("**Parsing example (first 8 rows)**")
        ex = df.head(8)[[core.SKILL_COL, "skills", "skill_count", "salary_range_usd", "salary_min", "salary_max", "salary_mid"]].copy()
        ex["skills"] = ex["skills"].map(lambda L: ", ".join(f"'{s}'" for s in L))
        st.dataframe(ex.rename(columns={core.SKILL_COL: "skills_required (raw)", "skills": "parsed list"}), hide_index=True)

    with t3:
        which = st.radio("Vocabulary", ["Required skills", "Preferred tools"], horizontal=True)
        col, pre = ("skills", "skill_") if which == "Required skills" else ("tools", "tool_")
        counts = core.skill_counts(df, col)
        fig = px.bar(counts.reset_index().rename(columns={"index": "Skill", 0: "Postings"}).set_axis(["Skill", "Postings"], axis=1),
                     x="Skill", y="Postings", color="Postings", color_continuous_scale="Viridis")
        fig.update_layout(coloraxis_showscale=False, height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        mh = core.multi_hot(df, col, pre)
        st.markdown(f"**Multi-hot matrix: {mh.shape[0]:,} rows × {mh.shape[1]} binary columns** (first 10 rows)")
        st.dataframe(pd.concat([df[[core.TARGET]].head(10), mh.head(10)], axis=1), hide_index=True)
        if col == "skills":
            st.markdown("**Skill co-occurrence** (how often two skills are required in the same posting)")
            co = pd.DataFrame(mh.values.T @ mh.values, index=mh.columns.str.replace("skill_", ""),
                              columns=mh.columns.str.replace("skill_", ""))
            np.fill_diagonal(co.values, 0)
            fig = px.imshow(co, aspect="auto", color_continuous_scale="Purples", labels=dict(color="Co-occurrences"))
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig)

    with t4:
        le = core.LabelEncoder().fit(df[core.TARGET])
        l, r = st.columns(2)
        with l:
            st.markdown("**Label encoding of the target (`job_title`)**")
            st.dataframe(pd.DataFrame({"Job role": le.classes_, "Code": range(len(le.classes_))}), hide_index=True)
        with r:
            st.markdown("**One-hot encoding of `experience_level`** (first 8 rows)")
            oh = pd.get_dummies(df[core.EXP_COL], prefix="exp").astype(int)
            st.dataframe(pd.concat([df[[core.EXP_COL]], oh], axis=1).head(8), hide_index=True)
        st.markdown("**StandardScaler on numeric inputs** (z = (x − mean) / std)")
        sc = core.StandardScaler().fit(df[core.NUM_FEATURES])
        scaled = pd.DataFrame(sc.transform(df[core.NUM_FEATURES]), columns=core.NUM_FEATURES)
        st.dataframe(pd.DataFrame({
            "feature": core.NUM_FEATURES, "mean": sc.mean_, "std": np.sqrt(sc.var_),
            "scaled mean": scaled.mean().values, "scaled std": scaled.std(ddof=0).values,
        }).round(3), hide_index=True)
        feat = st.selectbox("Distribution before / after scaling", core.NUM_FEATURES)
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x=feat, title="Before scaling", nbins=15))
        b.plotly_chart(px.histogram(scaled, x=feat, title="After scaling", nbins=15))
        st.caption("Inside the ML pipeline the scaler is fitted on the training split only (no data leakage).")

    with t5:
        fb = core.FeatureBuilder().fit(df[core.MODEL_COLS])
        Xm = pd.DataFrame(fb.transform(df[core.MODEL_COLS]), columns=fb.feature_names_)
        types = pd.Series(fb.feature_names_).map(core.feature_type).value_counts().rename_axis("Feature group").reset_index(name="Columns")
        c = st.columns([1, 2])
        c[0].markdown(f"**Model matrix: {Xm.shape[0]:,} × {Xm.shape[1]}**")
        c[0].dataframe(types, hide_index=True)
        c[1].dataframe(Xm.head(10).round(2), hide_index=True)
        out = pd.concat([df[[core.TARGET]], Xm], axis=1)
        st.download_button("⬇️ Download processed dataset (CSV)", out.to_csv(index=False).encode(), "processed_features.csv", "text/csv")


# =========================================================================== #
# PAGE 3 - EDA
# =========================================================================== #
def page_eda() -> None:
    st.title("📊 Interactive Exploratory Data Analysis")
    with st.expander("🔎 Filters", expanded=True):
        f = st.columns(4)
        sel_ind = f[0].multiselect("Industry", sorted(df["industry"].unique()))
        sel_exp = f[1].multiselect("Experience", core.EXP_ORDER)
        sel_size = f[2].multiselect("Company size", sorted(df["company_size"].unique()))
        sel_emp = f[3].multiselect("Employment type", sorted(df["employment_type"].unique()))
    d = df
    for col, sel in [("industry", sel_ind), (core.EXP_COL, sel_exp), ("company_size", sel_size), ("employment_type", sel_emp)]:
        if sel:
            d = d[d[col].isin(sel)]
    if len(d) < 20:
        st.warning("Too few rows after filtering — please relax the filters.")
        return

    k = core.dataset_kpis(d)
    c = st.columns(4)
    kpi(c[0], "Candidate profiles (postings)", f"{k['n_rows']:,}")
    kpi(c[1], "Unique job roles", str(k["n_roles"]))
    kpi(c[2], "Most in-demand skill", k["top_skill"], f"{k['top_skill_share']:.0%} of postings")
    kpi(c[3], "Average experience level", k["avg_exp_label"], f"avg {k['avg_skills']:.1f} skills / posting")
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["Top skills by role", "Skill × role heatmap", "Experience vs skill count",
                                  "Salary box plot", "More insights"])
    with t1:
        mode = st.radio("Show", ["Number of postings", "% of the role's postings"], horizontal=True)
        top10 = core.skill_counts(d).head(10).index.tolist()
        mh = core.multi_hot(d)[top10]
        mh[core.TARGET] = d[core.TARGET].values
        g = mh.groupby(core.TARGET)[top10]
        data = g.sum() if mode.startswith("Number") else g.mean() * 100
        long = data.reset_index().melt(id_vars=core.TARGET, var_name="Skill", value_name="Value")
        fig = px.bar(long, x="Skill", y="Value", color=core.TARGET, barmode="group",
                     category_orders={"Skill": top10},
                     labels={"Value": "Postings" if mode.startswith("Number") else "% of role postings", core.TARGET: "Job role"})
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("Top 10 skills by overall demand, split by job role.")
    with t2:
        m = core.role_skill_matrix(d)
        fig = px.imshow(m, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="Skill", y="Job role", color="Share of postings"))
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        spread = (m.max() - m.min()).sort_values(ascending=False)
        st.caption(f"Largest gap between roles for one skill: **{spread.index[0]}** ({spread.iloc[0] * 100:.1f} percentage points). "
                   "Small gaps mean skills barely separate the roles.")
    with t3:
        rng = np.random.default_rng(42)
        sc = d.copy()
        sc["Experience (jittered)"] = sc["exp_ord"] + rng.uniform(-0.22, 0.22, len(sc))
        sc["Skill count (jittered)"] = sc["skill_count"] + rng.uniform(-0.18, 0.18, len(sc))
        fig = px.scatter(sc, x="Experience (jittered)", y="Skill count (jittered)", color=core.TARGET, opacity=0.65,
                         hover_data={"experience_level": True, "skill_count": True, "Experience (jittered)": False,
                                     "Skill count (jittered)": False})
        fig.update_xaxes(tickvals=[0, 1, 2], ticktext=core.EXP_ORDER)
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("Experience level is categorical, so points are jittered to avoid overplotting.")
        st.dataframe(d.groupby(core.EXP_COL)["skill_count"].agg(["mean", "median", "count"]).reindex(core.EXP_ORDER).round(2).reset_index(),
                     hide_index=True)
    with t4:
        metric = st.selectbox("Salary measure", ["salary_mid", "salary_min", "salary_max"],
                              format_func=lambda x: {"salary_mid": "Midpoint", "salary_min": "Range minimum", "salary_max": "Range maximum"}[x])
        fig = px.box(d, x=core.TARGET, y=metric, color=core.EXP_COL, category_orders={core.EXP_COL: core.EXP_ORDER},
                     labels={metric: "USD", core.TARGET: "Job role", core.EXP_COL: "Experience"})
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("Salary distributions overlap heavily across roles and experience levels in this dataset.")
    with t5:
        a, b = st.columns(2)
        sx = d.explode("skills").merge(d[[core.TARGET]].reset_index(), left_index=True, right_on="index", how="left", suffixes=("", "_y")) if False else None
        # skill share by experience level
        mh2 = core.multi_hot(d)
        mh2[core.EXP_COL] = d[core.EXP_COL].values
        by_exp = mh2.groupby(core.EXP_COL).mean().reindex([e for e in core.EXP_ORDER if e in mh2[core.EXP_COL].unique()])
        a.markdown("**Skill demand by experience level**")
        a.plotly_chart(px.imshow(by_exp, text_auto=".0%", aspect="auto", color_continuous_scale="Purples"))
        ind = d.groupby("industry")["salary_mid"].median().sort_values().reset_index()
        b.markdown("**Median salary midpoint by industry**")
        b.plotly_chart(px.bar(ind, x="salary_mid", y="industry", orientation="h", labels={"salary_mid": "USD"}))
        st.markdown("**Skills per posting**")
        st.plotly_chart(px.histogram(d, x="skill_count", color=core.EXP_COL, barmode="group", nbins=6,
                                     category_orders={core.EXP_COL: core.EXP_ORDER}))


# =========================================================================== #
# PAGE 4 - ML models
# =========================================================================== #
def page_models() -> None:
    st.title("🤖 Machine Learning Models")
    st.caption("Task: predict the job role from the skills, preferred tools and experience level of a posting.")
    with st.expander("⚙️ Training settings"):
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
        seed = st.number_input("Random seed", 0, 9999, 42)
    bundle = get_models(DATA_KEY, df, float(test_size), int(seed))
    res, classes = bundle["results"], bundle["classes"]
    mt = core.metrics_table(bundle)
    real = [m for m in mt.index if not m.startswith("Baseline")]
    best = max(real, key=lambda m: mt.loc[m, "F1 (weighted)"])
    cv_col = [c for c in mt.columns if c.startswith("CV accuracy (")][0]

    c = st.columns(4)
    kpi(c[0], "Best model (weighted F1)", best, f"F1 {mt.loc[best, 'F1 (weighted)']:.3f}")
    kpi(c[1], "Test accuracy", f"{mt.loc[best, 'Accuracy']:.1%}", f"chance = {bundle['chance']:.1%}")
    kpi(c[2], f"{bundle['cv_folds']}-fold CV accuracy", f"{mt.loc[best, cv_col]:.1%}", f"± {mt.loc[best, 'CV accuracy std']:.1%}")
    kpi(c[3], "Train / test rows", f"{bundle['n_train']:,} / {bundle['n_test']:,}")
    st.write("")

    best_cv = max(mt.loc[m, cv_col] for m in real)
    if best_cv < bundle["chance"] + 0.05:
        st.warning(
            f"**Both models perform at roughly chance level** (best CV accuracy {best_cv:.1%} vs {bundle['chance']:.1%} for random guessing "
            f"among {len(classes)} roles). The statistical audit below shows the skills are almost independent of the job title in this "
            f"dataset. Report this honestly in your project: the pipeline is correct, but this data cannot support a reliable role classifier. "
            f"A dataset with real role-specific skill patterns (or real candidate labels) would be needed for useful predictions."
        )
    else:
        st.success("The models learn a meaningful signal beyond chance.")

    t1, t2, t3, t4, t5 = st.tabs(["Model comparison", "Confusion matrix", "Per-class metrics", "Feature importance", "Signal audit"])
    with t1:
        show = mt.copy()
        show["p-value vs baseline"] = [res[m]["p_vs_baseline"] for m in show.index]
        st.dataframe(show.round(4).reset_index().rename(columns={"index": "Model"}), hide_index=True)
        long = mt[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]].reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Metric", y="Score", color="index", barmode="group", labels={"index": "Model"})
        fig.add_hline(y=bundle["chance"], line_dash="dash", annotation_text="random guess")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("p-value vs baseline: one-sided binomial test of the test-set accuracy against always predicting the majority role. "
                   "A large p-value means no evidence the model beats the baseline.")
    with t2:
        model = st.selectbox("Model", real, key="cm_model")
        norm = st.toggle("Normalise by actual class (recall)", value=True)
        cm = res[model]["_cm"].astype(float)
        if norm:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig = px.imshow(cm, x=classes, y=classes, text_auto=".0%" if norm else ".0f", color_continuous_scale="Blues",
                        labels=dict(x="Predicted role", y="Actual role", color="Share" if norm else "Count"))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
    with t3:
        model = st.selectbox("Model", real, key="pc_model")
        st.dataframe(core.per_class_report(bundle, model).round(3), hide_index=True)
    with t4:
        model = st.selectbox("Model", real, key="fi_model")
        top_n = st.slider("Features to show", 5, 30, 15)
        fi = core.feature_importance(bundle, model).head(top_n)
        fig = px.bar(fi.iloc[::-1], x="importance", y="label", color="type", orientation="h",
                     labels={"importance": "Normalised importance", "label": "Feature", "type": "Group"})
        fig.update_layout(height=max(380, 24 * top_n), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("Random Forest: impurity-based importance. Logistic Regression: mean absolute coefficient across classes. "
                   "When a model is at chance, these rankings mostly reflect noise.")
    with t5:
        audit = get_audit(DATA_KEY, df)
        st.markdown(f"**{audit['n_significant']} of {audit['n_skills']} skills** are significantly linked to the job role "
                    "(chi-square test of independence, Benjamini–Hochberg FDR at 5%).")
        sk = audit["skills"].copy()
        sk["−log10 p"] = -np.log10(sk["p_value"].clip(lower=1e-300))
        fig = px.bar(sk, x="skill", y="−log10 p", color="significant (5%)")
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", annotation_text="p = 0.05 (unadjusted)")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.dataframe(sk.drop(columns="−log10 p").round(4), hide_index=True)
        st.markdown("**Other attributes vs job role**")
        st.dataframe(audit["other"].round(4), hide_index=True)


# =========================================================================== #
# PAGE 5 - Skills gap & recommendation
# =========================================================================== #
def page_recommender() -> None:
    st.title("🎯 Skills Gap & Job Recommendation")
    bundle = get_models(DATA_KEY, df, 0.2, 42)
    model_names = [m for m in bundle["results"] if not m.startswith("Baseline")]

    if "rec" not in st.session_state:
        st.session_state["rec"] = dict(
            skills=[s for s in ["Python", "SQL", "Pandas"] if s in VOCAB], tools=[], exp="Mid",
            role=ROLES[0], top_k=8, exp_only=False, model=model_names[0],
        )
    cur = st.session_state["rec"]

    with st.form("candidate_form"):
        st.markdown("**Your profile**")
        a, b = st.columns([2, 1])
        skills = a.multiselect("Current technical skills", VOCAB, default=cur["skills"])
        tools = a.multiselect("Tools you already use (optional)", TOOLS, default=cur["tools"])
        exp = b.selectbox("Experience level", core.EXP_ORDER, index=core.EXP_ORDER.index(cur["exp"]))
        role = b.selectbox("Desired target job role", ROLES, index=ROLES.index(cur["role"]))
        with st.expander("Advanced options"):
            o1, o2, o3 = st.columns(3)
            top_k = o1.slider("Skills that define a role (top-K)", 4, 12, cur["top_k"])
            exp_only = o2.checkbox("Use only postings at my experience level", value=cur["exp_only"],
                                   help="Smaller sample per role → noisier skill ranking.")
            model = o3.selectbox("Model used for confidence scores", model_names, index=model_names.index(cur["model"]))
        submitted = st.form_submit_button("🔍 Analyse my profile", type="primary")
    if submitted:
        st.session_state["rec"] = cur = dict(skills=skills, tools=tools, exp=exp, role=role, top_k=top_k,
                                             exp_only=exp_only, model=model)
    skills, tools, exp, role, top_k = cur["skills"], cur["tools"], cur["exp"], cur["role"], cur["top_k"]
    if not skills:
        st.info("Select at least one current skill and press **Analyse my profile**.")
        return

    profile, n_used, applied = core.role_profile(df, role, exp if cur["exp_only"] else None)
    gap = core.analyze_gap(skills, profile, OVERALL, top_k)
    matches = core.match_all_roles(skills, MATRIX, top_k)
    probs = core.predict_role_probs(bundle, cur["model"], core.candidate_frame(skills, tools, exp))
    best_match = matches.index[0]

    c = st.columns(4)
    kpi(c[0], f"Readiness for {role}", f"{gap['readiness']:.0%}", f"{len(gap['have'])} of {top_k} core skills")
    kpi(c[1], "Best-matching role (skills)", best_match, f"{matches.iloc[0]:.0%} match")
    kpi(c[2], f"Model confidence: {role}", f"{probs[role]:.1%}", f"chance = {bundle['chance']:.1%}")
    kpi(c[3], "Core skills missing", str(len(gap["missing"])), f"based on {n_used} postings" + (" at your level" if applied else ""))
    st.write("")

    if gap["readiness"] >= 0.75:
        st.success(f"You already cover most of what **{role}** postings ask for.")
    elif best_match != role and matches[best_match] - matches[role] > 0.05:
        st.info(f"Your skills currently overlap more with **{best_match}** ({matches[best_match]:.0%}) than with **{role}** ({matches[role]:.0%}).")

    st.markdown('<div class="section-title">1 · Role recommendation</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({"Skill-profile match %": matches * 100, "Model confidence %": probs.reindex(matches.index) * 100}).reset_index()
    comp = comp.rename(columns={"index": "Job role"}).melt(id_vars="Job role", var_name="Measure", value_name="Percent")
    fig = px.bar(comp, y="Job role", x="Percent", color="Measure", barmode="group", orientation="h",
                 category_orders={"Job role": list(matches.index[::-1])})
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.caption("**Skill-profile match**: share of a role's top-K skill demand that you cover. **Model confidence**: predicted probability "
               "from the classifier. Because roles share almost identical skill demand in this dataset, expect all scores to be close together.")

    st.markdown('<div class="section-title">2 · Skill gap: you vs the target role</div>', unsafe_allow_html=True)
    tbl = gap["table"]
    tr, tb = st.tabs(["Radar", "Bars"])
    with tr:
        theta = tbl["skill"].tolist()
        req = tbl["demand_pct"].tolist()
        you = [d if h else 0 for d, h in zip(req, tbl["have"])]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=req + req[:1], theta=theta + theta[:1], fill="toself", name=f"{role}: demand"))
        fig.add_trace(go.Scatterpolar(r=you + you[:1], theta=theta + theta[:1], fill="toself", name="You"))
        fig.update_layout(height=470, polar=dict(radialaxis=dict(visible=True, ticksuffix="%")), margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig)
    with tb:
        long = pd.DataFrame({
            "skill": list(tbl["skill"]) * 2,
            "Series": [f"{role}: demand (% of postings)"] * len(tbl) + ["You (demand covered)"] * len(tbl),
            "Percent": list(tbl["demand_pct"]) + [d if h else 0 for d, h in zip(tbl["demand_pct"], tbl["have"])],
        })
        fig = px.bar(long, x="skill", y="Percent", color="Series", barmode="group", category_orders={"skill": list(tbl["skill"])})
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
    st.caption("Radar values = how often the skill appears in this role's postings; your line shows that value only where you have the skill.")

    st.markdown('<div class="section-title">3 · Missing skills report</div>', unsafe_allow_html=True)
    h, m = st.columns(2)
    h.markdown("**You already have**<br>" + pills(gap["have"]["skill"].tolist(), "ok"), unsafe_allow_html=True)
    m.markdown("**Missing for this role**<br>" + pills(gap["missing"]["skill"].tolist(), "miss"), unsafe_allow_html=True)
    if gap["extra"]:
        st.caption("Other skills you listed (not in this role's top-K): " + ", ".join(gap["extra"]))
    if len(gap["missing"]):
        rep_df = gap["missing"].copy()
        rep_df["Suggested certification / course"] = rep_df["skill"].map(
            lambda s: (core.SKILL_META.get(s, {}).get("cert", "-") if core.SKILL_META.get(s, {}).get("cert", "-") != "-"
                       else core.SKILL_META.get(s, {}).get("learn", "Official docs + small project")))
        st.dataframe(
            rep_df[["priority", "skill", "demand_pct", "lift", "Suggested certification / course"]].rename(
                columns={"priority": "Priority", "skill": "Skill", "demand_pct": "Demand (% of postings)", "lift": "Lift vs all roles"}),
            hide_index=True,
            column_config={
                "Demand (% of postings)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                "Lift vs all roles": st.column_config.NumberColumn(format="%.2f×"),
            },
        )
        st.caption("Lift > 1 means the skill is requested more often by this role than average. The certification column is curated "
                   "guidance (not in the dataset) — verify current availability before enrolling.")
    else:
        st.success("No gaps among the top skills for this role.")

    st.markdown('<div class="section-title">4 · Personalised learning roadmap</div>', unsafe_allow_html=True)
    road = core.build_roadmap(gap["missing"], role)
    if road["phases"]:
        st.markdown(f"Estimated effort: **~{road['total_weeks']} weeks** (rough guide for part-time study).")
        for ph in road["phases"]:
            with st.expander(f"{ph['name']} — ~{ph['weeks']} weeks", expanded=True):
                for s in ph["steps"]:
                    cert = f" · 🎓 {s['cert']}" if s["cert"] != "-" else ""
                    st.markdown(f"- **{s['skill']}** ({s['priority']}, ~{s['weeks']} wk): {s['learn']}{cert}")
    else:
        st.write("No gaps found — spend the time on portfolio work.")
    st.info(f"**Capstone project:** {road['project']}")
    st.download_button("⬇️ Download roadmap (Markdown)", core.roadmap_markdown(role, gap["readiness"], road).encode(),
                       f"roadmap_{role.replace(' ', '_')}.md", "text/markdown")

    sal = df[(df[core.TARGET] == role) & (df[core.EXP_COL] == exp)]["salary_mid"]
    if len(sal) >= 10:
        st.caption(f"Context: median advertised salary midpoint for {exp}-level {role} postings is ${sal.median():,.0f} (n={len(sal)}). "
                   "Salary differences between roles are not statistically significant in this dataset.")


PAGES = {NAV[0]: page_overview, NAV[1]: page_preprocessing, NAV[2]: page_eda, NAV[3]: page_models, NAV[4]: page_recommender}
PAGES[page]()
