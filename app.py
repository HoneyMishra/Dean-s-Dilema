"""The Dean's Dilemma - interactive Streamlit companion to the deck."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an

st.set_page_config(page_title="The Dean's Dilemma", page_icon="🎓", layout="wide")

NAVY, TEAL, AMBER, RED, GREY = "#1F4E79", "#2A9D8F", "#E9A23B", "#C8553D", "#8A94A6"


@st.cache_data
def get_data():
    return an.load_data()


@st.cache_data
def get_rq1(df):
    return an.rq1_odds_ratios(df)


@st.cache_data
def get_rq2(df):
    return an.rq2_models(df)


@st.cache_data
def get_gender(df):
    return an.gender_stats(df)


@st.cache_data(show_spinner="Running cross-validation (about a minute the first time)...")
def get_cv(df, repeats):
    return an.cv_models(df, 5, repeats)


@st.cache_resource
def get_risk_model(_df):
    return an.fit_risk_model(_df)


df = get_data()
placed = df[df.Placed == 1]
fmt_l = lambda v: f"₹{v/1e5:.2f}L"


def pct(x, d=1):
    return f"{x*100:.{d}f}%"


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white", font=dict(size=13))
    fig.update_xaxes(gridcolor="#E6EAF0")
    fig.update_yaxes(gridcolor="#E6EAF0")
    return fig


def answer(text):
    st.success(f"**Answer:** {text}")


PAGES = ["The Dilemma", "Data & Approach", "EDA: Outcome & Salary", "EDA: Who Gets Placed?",
         "EDA: Student Segments", "RQ1 · Entrance Test", "RQ2 · Academic Drivers",
         "RQ3 · Cut-offs", "RQ6 · Gender", "RQ7 · Early Warning", "Conclusion"]

with st.sidebar:
    st.title("🎓 The Dean's Dilemma")
    st.caption("Who should we admit, and what does it mean for placement?")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Synthetic data: 5,000 MBA students, 65/35 placed split by design.")
    with open("docs/Dean_Dilemma_Final.pptx", "rb") as f:
        st.download_button("Download the deck (.pptx)", f, "Dean_Dilemma_Final.pptx",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation")

# ---------------------------------------------------------------- Dilemma
if page == PAGES[0]:
    st.title("The Dean's Dilemma")
    st.subheader("The Dean must set admission policy, with 35% of students unplaced")
    st.write("Placement drives the institute's reputation, yet 1 in 3 students "
             f"({(df.Placed==0).sum():,} of {len(df):,}) leave unplaced. The Dean must decide which "
             "admission levers actually improve outcomes, without bias and without shrinking intake unnecessarily.")
    c = st.columns(4)
    c[0].metric("Students", f"{len(df):,}")
    c[1].metric("Placement rate", pct(df.Placed.mean()))
    c[2].metric("Median salary (placed)", fmt_l(placed.Salary.median()))
    c[3].metric("Unplaced", f"{(df.Placed==0).sum():,}")
    st.markdown("#### Research questions")
    rqs = [("RQ1", "Should the entrance test be mandatory? Is a minimum percentile justified?"),
           ("RQ2", "Which academic variables truly matter, for placement and for salary?"),
           ("RQ3", "Where should cut-offs on entrance percentile and SSC % be set?"),
           ("RQ6", "Is there a gender salary gap, or a placement-access gap?"),
           ("RQ7", "Can admission data flag at-risk students early (an early-warning model)?")]
    for k, q in rqs:
        st.markdown(f"**{k}** &nbsp; {q}")
    st.info("Each research question maps to one concrete decision the Dean has to take. "
            "Gender is analysed for equity, never as an admission criterion.")

# ---------------------------------------------------------------- Data
elif page == PAGES[1]:
    st.title("Data & Approach")
    st.subheader("5,000 students, 26 variables, clean and internally consistent")
    c = st.columns(3)
    c[0].metric("Students", "5,000", "65% placed / 35 % not")
    c[1].metric("Variables", "26", "academics, test, MBA marks, outcome")
    c[2].metric("Placement flag = Placement_B", "100%")
    flag_ok = ((df.Placement == "Placed").astype(int) == df.Placement_B).mean()
    zero_et = int((df.Test_Taken & (df.Percentile_ET == 0)).sum())
    l, r = st.columns(2)
    with l:
        st.markdown("##### Data preparation")
        st.markdown(f"- {zero_et} test-takers had Percentile_ET = 0, treated as missing, not as a zero score\n"
                    "- Entrance score grouped: 75+, 50–75, Below 50, No test, Missing score\n"
                    "- Duplicate dummy columns dropped; log-salary for placed students only\n"
                    "- No out-of-range percentages; no salary for unplaced students")
        st.caption(f"Live check: Placement vs Placement_B agreement = {flag_ok:.0%}; "
                   f"unplaced students with salary > 0: {int(((df.Placed==0)&(df.Salary>0)).sum())}.")
    with r:
        st.markdown("##### Methods used")
        st.markdown("- Chi-square, t-test, Mann-Whitney for group differences\n"
                    "- Logistic regression (odds ratios) and OLS on log-salary\n"
                    "- Cut-off curves with knee detection; Wilson & bootstrap CIs\n"
                    "- Neumark-style gap decomposition; K-Means (k=4)\n"
                    "- Models: Logistic, Bagging, Random Forest, Gradient Boosting (repeated CV)")
    st.markdown("##### Browse the data")
    st.dataframe(df.drop(columns=["Placed", "Test_Taken", "ET_Score", "Gender_Label"]),
                 width="stretch", height=300)

# ---------------------------------------------------------------- EDA outcome
elif page == PAGES[2]:
    st.title("EDA | Outcome & Salary")
    st.subheader("Two-thirds are placed; salaries cluster tightly around ₹2.6 lakh")
    l, r = st.columns(2)
    with l:
        v = df.Placement.value_counts().reset_index()
        fig = px.pie(v, names="Placement", values="count", hole=0.55,
                     color="Placement", color_discrete_map={"Placed": TEAL, "Not Placed": RED})
        fig.update_layout(title="Placement outcome")
        st.plotly_chart(style(fig), width="stretch")
    with r:
        scale = st.radio("Salary scale", ["Rupees", "Log salary"], horizontal=True)
        x = placed.Salary if scale == "Rupees" else np.log(placed.Salary)
        fig = px.histogram(x=x, nbins=50, color_discrete_sequence=[NAVY])
        fig.update_layout(title="Salary distribution (placed students)", xaxis_title=scale, yaxis_title="Students")
        st.plotly_chart(style(fig), width="stretch")
    st.markdown("**Observation**")
    st.markdown(f"- Overall placement is **{pct(df.Placed.mean())}** ({int(df.Placed.sum()):,} placed).\n"
                f"- Median salary **{fmt_l(placed.Salary.median())}**, mean **{fmt_l(placed.Salary.mean())}**, "
                f"max **{fmt_l(placed.Salary.max())}** → right-skewed, so log-salary is used in models.")

# ---------------------------------------------------------------- EDA who
elif page == PAGES[3]:
    st.title("EDA | Who Gets Placed?")
    st.subheader("Placement differs sharply by entrance-test status, SSC and experience")
    dim = st.selectbox("Break placement down by", ["Entrance_Group", "Gender", "Board_SSC", "Stream_HSC",
                                                   "Course_Degree", "Specialization_MBA", "Experience_Yrs"])
    g = an.placement_by_group(df, dim)
    fig = px.bar(g, x=dim, y="rate", text=g["rate"].map(pct), color_discrete_sequence=[NAVY],
                 error_y=g["hi"] - g["rate"], error_y_minus=g["rate"] - g["lo"])
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Placement rate (95% Wilson CI)")
    fig.add_hline(y=df.Placed.mean(), line_dash="dash", line_color=GREY, annotation_text="overall")
    st.plotly_chart(style(fig), width="stretch")
    l, r = st.columns(2)
    with l:
        from scipy import stats
        rows = []
        for c in ["Percent_SSC", "Percent_HSC", "Percent_Degree", "Experience_Yrs", "ET_Score",
                  "Percent_MBA", "Marks_Communication", "Marks_Projectwork", "Marks_BOCA"]:
            a = df[df.Placed == 1][c].dropna(); b = df[df.Placed == 0][c].dropna()
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({"Variable": c, "Mean (placed)": a.mean(), "Mean (not placed)": b.mean(), "p-value": p})
        st.markdown("##### t-tests: placed vs not placed")
        st.dataframe(pd.DataFrame(rows).style.format({"Mean (placed)": "{:.2f}", "Mean (not placed)": "{:.2f}",
                                                      "p-value": "{:.3g}"}), hide_index=True, width="stretch")
    with r:
        ct = pd.crosstab(df["Entrance_Group"], df["Placed"])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        st.markdown("##### Statistical screening")
        st.markdown(f"- Chi-square (placement vs entrance group): **χ² = {chi2:.0f}, p < .001**, by far the strongest\n"
                    "- Gender, SSC board, HSC stream, degree course and specialization all significant (p < .01)\n"
                    "- Communication marks do **not** differ (p = 0.18)")
        st.info("Takeaway: entrance-test status is the dominant separator → RQ1 and RQ3 test it directly.")

# ---------------------------------------------------------------- segments
elif page == PAGES[4]:
    st.title("EDA | Student Segments")
    st.subheader("K-Means finds four profiles; placement ranges from 45% to 74%")
    seg = pd.DataFrame({
        "Segment": ["0  High achievers", "3  Experienced mid", "2  Fresher mid", "1  Low test score"],
        "Students": [1669, 1186, 1649, 496], "Mean SSC %": [73.4, 59.7, 58.5, 61.6],
        "Mean entrance": [72.2, 67.4, 68.2, 38.4], "Mean exp. (yrs)": [0.36, 1.31, 0.00, 0.29],
        "Placement": [0.741, 0.689, 0.590, 0.450], "Median salary (₹L)": [2.75, 2.51, 2.50, 2.50]})
    st.caption("Segment profiles as reported in the notebook analysis behind the deck.")
    st.dataframe(seg.style.format({"Students": "{:,}", "Placement": "{:.1%}", "Mean SSC %": "{:.1f}",
                                   "Mean entrance": "{:.1f}", "Mean exp. (yrs)": "{:.2f}",
                                   "Median salary (₹L)": "₹{:.2f}L"}), hide_index=True, width="stretch")
    fig = px.bar(seg, x="Segment", y="Placement", text=seg["Placement"].map(pct),
                 color_discrete_sequence=[NAVY])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(fig, 330), width="stretch")
    st.markdown("**Interpretation**\n"
                "- Segment 1 has decent SSC (61.6) but a very low entrance score (38), and only 45% get placed: "
                "the entrance score, not school marks, marks the at-risk group.\n"
                "- Segments 2 vs 3 have near-identical marks, yet prior experience lifts placement ~10 points (59% → 69%).\n"
                "- A depth-3 decision tree splits on entrance score (~50), HSC % and degree %, consistent with the RQ results ahead.")

# ---------------------------------------------------------------- RQ1
elif page == PAGES[5]:
    st.title("RQ1 | Entrance Test")
    st.subheader("A score below 50 cuts placement to 43%, but a test waiver costs far less")
    g = an.placement_by_group(df, "Entrance_Group")
    fig = px.bar(g, x="Entrance_Group", y="rate", text=g["rate"].map(pct), color_discrete_sequence=[NAVY],
                 error_y=g["hi"] - g["rate"], error_y_minus=g["rate"] - g["lo"])
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Placement rate (95% Wilson CI)")
    fig.update_xaxes(title="Entrance score group")
    st.plotly_chart(style(fig), width="stretch")
    orr = get_rq1(df)
    ent = orr.loc[["50–75", "No test", "Below 50", "Missing score"]]
    st.markdown("##### Adjusted odds ratios vs 75+ group (controls: SSC %, degree %, experience)")
    fig = go.Figure(go.Scatter(x=ent.OR, y=ent.index, mode="markers", marker=dict(size=11, color=NAVY),
                               error_x=dict(type="data", symmetric=False, array=ent.hi - ent.OR,
                                            arrayminus=ent.OR - ent.lo)))
    fig.add_vline(x=1, line_dash="dash", line_color=GREY)
    fig.update_xaxes(title="Odds ratio (log scale)", type="log")
    st.plotly_chart(style(fig, 280), width="stretch")
    st.dataframe(ent.style.format({"OR": "{:.2f}", "lo": "{:.2f}", "hi": "{:.2f}", "p": "{:.1e}"}),
                 width="stretch")
    r = g.set_index("Entrance_Group")
    st.markdown("**Analysis & interpretation**")
    st.markdown(f"- Below-50 students place at **{pct(r.loc['Below 50','rate'])}** "
                f"(CI {pct(r.loc['Below 50','lo'])}–{pct(r.loc['Below 50','hi'])}) vs **{pct(r.loc['50–75','rate'])}** "
                f"for 50–75: a ~25-point deficit that survives controls (OR {ent.loc['Below 50','OR']:.2f}).\n"
                f"- Placement rises step-wise with score (43% → 68% → 79%): the test carries real signal.\n"
                f"- No-test students ({pct(r.loc['No test','rate'])}) sit 17 points above Below-50 and ~7 below 50–75: "
                "a modest, not catastrophic, penalty.")
    answer("a minimum percentile (~50) is justified; a blanket \"test mandatory for all\" is not required.")

# ---------------------------------------------------------------- RQ2
elif page == PAGES[6]:
    st.title("RQ2 | Academic Drivers")
    st.subheader("SSC %, entrance score and experience drive placement, not salary")
    odds, sal, r2, vif = get_rq2(df)
    names = {"Percent_SSC": "SSC %", "Percent_HSC": "HSC %", "Percent_Degree": "Degree %",
             "Experience_Yrs": "Experience", "ET_Score": "Entrance score", "ET_Missing": "Entrance score missing"}
    l, r = st.columns(2)
    with l:
        o = odds.rename(index=names).sort_values("OR")
        fig = px.bar(o, x="OR", y=o.index, orientation="h", text=o.OR.map(lambda v: f"{v:.2f}"),
                     color_discrete_sequence=[NAVY])
        fig.add_vline(x=1, line_dash="dash", line_color=GREY)
        fig.update_layout(title="Placement: odds ratio per +1 SD (logistic)")
        st.plotly_chart(style(fig), width="stretch")
    with r:
        s = sal.rename(index=names).sort_values("pct_per_SD")
        fig = px.bar(s, x="pct_per_SD", y=s.index, orientation="h", text=s.pct_per_SD.map(lambda v: f"{v:+.1f}%"),
                     color=s.p < 0.05, color_discrete_map={True: TEAL, False: GREY})
        fig.update_layout(title=f"Salary: % change per +1 SD (OLS log-salary, R² = {r2:.2f})", showlegend=False)
        st.plotly_chart(style(fig), width="stretch")
    st.caption("Teal = significant at 5%. Salary model uses placed students only.")
    st.markdown("**Analysis & interpretation**")
    st.markdown(f"- SSC % is the strongest placement predictor: +1 SD raises the odds of placement **{odds.loc['Percent_SSC','OR']:.2f}×**.\n"
                f"- Entrance score ({odds.loc['ET_Score','OR']:.2f}×) and experience ({odds.loc['Experience_Yrs','OR']:.2f}×) add independent value; "
                f"missing score carries a penalty (OR {odds.loc['ET_Missing','OR']:.2f}).\n"
                f"- Degree % is slightly negative once others are controlled (OR {odds.loc['Percent_Degree','OR']:.2f}) and HSC % is not significant.\n"
                f"- No multicollinearity (max VIF {vif.max():.2f}), so coefficients are reliable.\n"
                f"- Academics explain who is placed, not how much they earn (R² {r2:.0%}).")
    answer("screen on SSC %, entrance score and experience; degree % and HSC % add little.")

# ---------------------------------------------------------------- RQ3
elif page == PAGES[7]:
    st.title("RQ3 | Cut-offs")
    st.subheader("Returns to higher cut-offs are real but gradual; there is no sharp threshold")
    st.caption("Slide the cut-off and see how many students you keep and how placement changes.")
    c1, c2 = st.columns(2)
    for col_, label, col, lo, hi, default, note in [
            (c1, "Entrance percentile", "ET_Score", 20, 95, 50, "Usable scores only (test-takers with a valid percentile)."),
            (c2, "SSC %", "Percent_SSC", 40, 90, 58, "All students.")]:
        with col_:
            st.markdown(f"##### {label}")
            cut = st.slider(f"Minimum {label}", lo, hi, default, key=col)
            curve, base_rate = an.cutoff_curve(df, col, list(range(lo, hi + 1)))
            sel = curve[curve.cutoff == cut].iloc[0]
            m = st.columns(3)
            m[0].metric("Intake kept", pct(sel.kept, 0))
            m[1].metric("Placement", pct(sel.placement), f"{(sel.placement-base_rate)*100:+.1f} pts")
            m[2].metric("Baseline", pct(base_rate))
            fig = go.Figure()
            fig.add_scatter(x=curve.cutoff, y=curve.placement, name="Placement rate", line=dict(color=NAVY, width=3))
            fig.add_scatter(x=curve.cutoff, y=curve.kept, name="Share of intake kept", line=dict(color=AMBER, width=3))
            fig.add_vline(x=cut, line_dash="dash", line_color=GREY)
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_xaxes(title=f"Cut-off on {label}")
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(style(fig, 360), width="stretch")
            st.caption(note)
    st.markdown("**Reading the curves**")
    st.markdown("- **Entrance percentile:** cut-off 50 keeps 84% of test-takers, placement 67.6% → 72.4% (+4.7 pts). "
                "The mathematical knee is 90 (keeps only 10%), which is impractical.\n"
                "- **SSC %:** cut-off 56–60 keeps 62–75% of intake, placement 65% → 70–72%. "
                "Knee at 82 keeps just 4.9% of intake, also impractical.")
    answer("use soft floors, entrance ≥ 50 and SSC ≈ 56–60; steeper cut-offs gain few points for large intake loss.")

# ---------------------------------------------------------------- RQ6
elif page == PAGES[8]:
    st.title("RQ6 | Gender")
    st.subheader("No salary gap, but women are placed 6 points less often")
    gs = get_gender(df)
    n = df.groupby("Gender").agg(n=("Placed", "size"), placed=("Placed", "sum"))
    c = st.columns(4)
    c[0].metric("Male / Female", f"{n.loc['M','n']:,} / {n.loc['F','n']:,}")
    c[1].metric("Placed (M / F)", f"{n.loc['M','placed']:,} / {n.loc['F','placed']:,}")
    c[2].metric("Placement M vs F", f"{pct(n.loc['M','placed']/n.loc['M','n'])} vs {pct(n.loc['F','placed']/n.loc['F','n'])}")
    c[3].metric("Salary gap (M − F)", f"₹{-gs['gap_F_minus_M']:,.0f}")
    l, r = st.columns(2)
    with l:
        g = an.placement_by_group(df, "Gender_Label")
        fig = px.bar(g, x="Gender_Label", y="rate", text=g["rate"].map(pct), color_discrete_sequence=[NAVY],
                     error_y=g["hi"] - g["rate"], error_y_minus=g["rate"] - g["lo"])
        fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Placement rate")
        fig.update_xaxes(title="")
        fig.update_layout(title="Placement access")
        st.plotly_chart(style(fig), width="stretch")
    with r:
        fig = px.box(placed, x="Gender_Label", y="Salary", color="Gender_Label",
                     color_discrete_map={"Male": NAVY, "Female": TEAL})
        fig.update_layout(title="Salary (placed students)", showlegend=False, xaxis_title="")
        st.plotly_chart(style(fig), width="stretch")
    st.markdown("**Analysis & interpretation**")
    st.markdown(f"- Raw gap (M − F) is **₹{-gs['gap_F_minus_M']:,.0f}** (women earn slightly more), p = {gs['p_salary']:.2f}; "
                f"bootstrap 95% CI ₹{-gs['ci'][1]:,.0f} to ₹{-gs['ci'][0]:,.0f} includes zero.\n"
                "- With controls the female coefficient stays ~0 (+1.0%, p = 0.35); unchanged excluding salary outliers.\n"
                f"- The real disparity is **placement access**: 60.8% vs 67.1% (χ² = {gs['chi2']:.1f}, p < .001).\n"
                "- Gender is never an admission criterion.")
    answer("no pay penalty to fix; investigate recruiter mix and placement-cell access for women.")

# ---------------------------------------------------------------- RQ7
elif page == PAGES[9]:
    st.title("RQ7 | Early Warning")
    st.subheader("Early warning works: Gradient Boosting reaches AUC 0.83")
    st.caption("Target = *not placed* (the at-risk class). Features: SSC, HSC, degree %, experience, entrance "
               "score and test status. Gender excluded.")
    reps = st.select_slider("Cross-validation repeats (5-fold × repeats)", [1, 2, 5], value=2,
                            help="The deck used 5×5. More repeats = slower first run.")
    cv = get_cv(df, reps)
    l, r = st.columns(2)
    with l:
        fig = px.bar(cv, x="Model", y="AUC", error_y="AUC_sd", text=cv.AUC.map(lambda v: f"{v:.2f}"),
                     color_discrete_sequence=[NAVY])
        fig.update_yaxes(range=[0.4, 0.95])
        fig.update_layout(title="ROC AUC (higher is better)")
        st.plotly_chart(style(fig), width="stretch")
    with r:
        fig = px.bar(cv, x="Model", y="Recall (Not placed)", text=cv["Recall (Not placed)"].map(lambda v: f"{v:.2f}"),
                     color_discrete_sequence=[AMBER])
        fig.update_layout(title="Recall on Not-placed class")
        st.plotly_chart(style(fig), width="stretch")
    st.dataframe(cv.style.format({c: "{:.3f}" for c in cv.columns if c != "Model"}), hide_index=True,
                 width="stretch")
    gb = cv.set_index("Model").loc["Gradient Boosting"]
    lg = cv.set_index("Model").loc["Logistic"]
    st.info("**Deck figures** (notebook, 5×5 CV): Gradient Boosting AUC 0.829 ± 0.011, "
            "recall 0.444; Logistic ≈ 0.68; tree ensembles 0.78–0.83. The live table above uses a simplified "
            "feature set listed here, so its numbers can differ from the notebook, though the story is the same.")
    st.markdown("**Analysis & interpretation**")
    st.markdown("- Majority-class baseline: AUC 0.50 and recall 0; it never finds an unplaced student.\n"
                f"- Gradient Boosting (live): AUC {gb.AUC:.2f}, recall {gb['Recall (Not placed)']:.2f} at the default threshold.\n"
                f"- Logistic (live AUC {lg.AUC:.2f}) vs tree ensembles: non-linear interactions matter, as segmentation hinted.\n"
                "- Recall well below 1 means many at-risk students are missed at the default threshold; "
                "tune the threshold for support triage.\n- Exclude gender from any operational model.")
    st.markdown("---")
    st.markdown("##### Try it: at-risk score for a hypothetical applicant")
    mdl, cols, imp = get_risk_model(df)
    a, b, c3 = st.columns(3)
    ssc = a.slider("SSC %", 35, 95, 60); hsc = a.slider("HSC %", 35, 95, 60)
    deg = b.slider("Degree %", 35, 95, 60); exp = b.slider("Experience (yrs)", 0, 3, 0)
    tt = c3.selectbox("Entrance test", ["Took test, score known", "Took test, score missing", "No test"])
    et = c3.slider("Entrance percentile", 1, 99, 60, disabled=tt != "Took test, score known")
    thr = st.slider("Flag threshold (lower = catch more at-risk students)", 0.1, 0.7, 0.35, 0.05)
    row = pd.DataFrame([{"Percent_SSC": ssc, "Percent_HSC": hsc, "Percent_Degree": deg, "Experience_Yrs": exp,
                         "ET_Missing": int(tt != "Took test, score known"),
                         "ET_Score": et if tt == "Took test, score known" else df.ET_Score.median(),
                         "Test_Taken_I": int(tt != "No test")}])[cols]
    p = mdl.predict_proba(row)[0, 1]
    m = st.columns(2)
    m[0].metric("Predicted risk of NOT being placed", pct(p, 0))
    m[1].metric("Support flag", "⚠️ Flag for support" if p >= thr else "✅ Not flagged")
    st.caption("A support-targeting tool, not an admission filter.")
    fig = px.bar(imp, x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[NAVY])
    fig.update_layout(title="Feature importance (Gradient Boosting)", xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(style(fig, 300), width="stretch")
    answer("yes, build the early-warning model as a support-targeting tool, not as an admission filter.")

# ---------------------------------------------------------------- Conclusion
else:
    st.title("Final Conclusion")
    st.subheader("Admit on evidence, support on prediction, and fix access, not pay")
    concl = pd.DataFrame([
        ["RQ1 Entrance test", "Below 50 → 43% placed vs 68% (OR 0.24)", "Set a ~50th-percentile floor; test not needed as a blanket gate"],
        ["RQ2 Key variables", "SSC OR 1.57, entrance 1.48, experience 1.20", "Screen on SSC %, entrance, experience; drop degree/HSC %"],
        ["RQ3 Cut-offs", "Gradual gains; knees at 90 / 82 keep ≤10%", "Soft floors: entrance ≥ 50, SSC ≈ 56–60"],
        ["RQ6 Gender", "Salary gap ≈ 0 (p = 0.48); placement 60.8% vs 67.1%", "Widen recruiter access for women; no pay fix needed"],
        ["RQ7 Early warning", "GB AUC 0.83, recall 0.44 vs baseline 0.50 / 0", "Deploy to target support for at-risk segments; tune threshold"]],
        columns=["Decision", "Evidence", "Recommendation"])
    st.dataframe(concl, hide_index=True, width="stretch")
    st.markdown("#### The Dean can act on three levers")
    c = st.columns(3)
    c[0].info("**1. Soft floors**\n\nEntrance ≥ 50, SSC ≈ 56–60")
    c[1].info("**2. Early-warning support**\n\nTarget at-risk students with the GB model")
    c[2].info("**3. Placement access for women**\n\nWiden recruiter mix; no pay fix needed")
    st.warning("**Limitations:** synthetic data with a designed 65/35 split; results are associations, not causal; "
               "entrance-score zeros treated as missing; salary R² is only 6%, so pay is driven by factors "
               "outside admission data.")
