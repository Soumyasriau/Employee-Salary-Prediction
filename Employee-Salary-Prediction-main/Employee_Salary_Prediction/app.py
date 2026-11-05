import os
import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = "linearmodel.pkl"
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"

JOB_LEVELS = {
    1: "Intern",
    2: "Junior",
    3: "Mid-level",
    4: "Senior",
    5: "Lead",
}

# Monthly base salary for each level (in rupees)
MIN_SALARY = {
    1: 10000,
    2: 15000,
    3: 25000,
    4: 40000,
    5: 60000,
}

# Every year of experience adds this share of the level base (capped at 20 years)
EXP_GROWTH_PER_YEAR = 0.08
EXP_YEARS_CAP = 20

# Multipliers: edit these numbers to tune the estimate.
# They are assumptions, not learned from data.
COUNTRY_MULT = {
    "India": 1.0, "United States": 3.0, "United Kingdom": 2.3,
    "Canada": 2.1, "Australia": 2.2, "Germany": 2.3, "Other": 1.2,
}
EDUCATION_MULT = {
    "Bachelor's degree": 1.00, "Master's degree": 1.10,
    "Doctoral degree": 1.20, "College diploma": 0.95, "Other": 0.90,
}
DEVTYPE_MULT = {
    "Software Developer": 1.00, "Web Developer": 0.95, "Data Scientist": 1.15,
    "Mobile Developer": 1.05, "DevOps Engineer": 1.10, "Other": 0.95,
}
WORK_MULT = {
    "Full-time": 1.00, "Part-time": 0.55, "Freelance": 1.20,
    "Contract": 1.10, "Not currently working": 0.30,
}
INDUSTRY_MULT = {
    "Technology": 1.10, "Finance": 1.15, "Healthcare": 1.00,
    "Education": 0.85, "E-commerce": 1.00, "Other": 0.95,
}

st.set_page_config(
    page_title="Developer Salary Predictor",
    page_icon="💰",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #fff7fb 0%, #f7f5ff 50%, #f8fafc 100%);
    color: #202033;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #fff0f6, #f4efff);
    border: 1px solid #f0dced;
    border-radius: 24px;
    padding: 30px 34px;
    margin-bottom: 24px;
}

.eyebrow {
    color: #b83280;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero h1 {
    color: #27213c;
    font-size: clamp(1.8rem, 4vw, 2.7rem);
    line-height: 1.12;
    margin: 0 0 10px;
}

.hero p {
    color: #716b82;
    margin: 0;
    font-size: 1rem;
}

.section-title {
    color: #29233e;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 8px 0 4px;
}

.section-sub {
    color: #777188;
    margin-bottom: 18px;
}

[data-testid="stForm"] {
    background: white;
    border: 1px solid #eee7f3;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(53, 36, 78, .06);
}

label, [data-testid="stWidgetLabel"] p {
    color: #49435d !important;
    font-weight: 600 !important;
}

div.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #d9468f, #8b5cf6);
    color: white;
    border: 0;
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
}

div.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    filter: brightness(.96);
}

.result-card {
    background: linear-gradient(135deg, #fff0f6, #f1edff);
    border: 1px solid #e8d5f3;
    border-radius: 22px;
    padding: 26px;
    margin-top: 22px;
}

.result-label {
    color: #8e3a83;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.salary {
    color: #30264b;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    line-height: 1.15;
    margin: 8px 0;
}

.result-note {
    color: #746887;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #eee7f3;
    border-radius: 16px;
}

.footnote {
    color: #817b91;
    font-size: .82rem;
    margin-top: 22px;
}

@media (max-width: 640px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }
    .hero {
        padding: 22px 18px;
    }
    [data-testid="stForm"] {
        padding: 18px 14px;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, MODEL_FILE))

    sx_path = os.path.join(BASE_DIR, SCALER_X_FILE)
    sy_path = os.path.join(BASE_DIR, SCALER_Y_FILE)

    scaler_x = joblib.load(sx_path) if os.path.exists(sx_path) else None
    scaler_y = joblib.load(sy_path) if os.path.exists(sy_path) else None

    return model, scaler_x, scaler_y


def predict_salary(model, scaler_x, scaler_y, years):
    features = np.array([[years]], dtype=float)

    if scaler_x is not None:
        features = scaler_x.transform(features)

    prediction = model.predict(features)

    if scaler_y is not None:
        prediction = scaler_y.inverse_transform(
            np.array(prediction).reshape(-1, 1)
        )

    return float(np.ravel(prediction)[0])


def estimate_monthly(raw, years, age, level, country, education,
                     developer_type, work_history, industry):
    """Combine the model output with every profile field."""
    level_base = MIN_SALARY[level]

    # Experience adds on top of the level base, so 0, 1, 2, 3... years all differ
    exp_bonus = level_base * EXP_GROWTH_PER_YEAR * min(years, EXP_YEARS_CAP)

    # Use the model prediction if it is higher than the level base
    core = max(raw, level_base) + exp_bonus

    # Small age effect: older candidates (beyond 22) earn a little more, capped
    age_mult = 1 + min(max(age - 22, 0), 20) * 0.003

    multiplier = (
        COUNTRY_MULT[country]
        * EDUCATION_MULT[education]
        * DEVTYPE_MULT[developer_type]
        * WORK_MULT[work_history]
        * INDUSTRY_MULT[industry]
        * age_mult
    )

    return core * multiplier, level_base, exp_bonus, core, multiplier


st.markdown("""
<section class="hero">
    <div class="eyebrow">Career insights · Salary estimate</div>
    <h1>💰 Developer Salary Predictor</h1>
    <p>Explore an estimated salary based on your developer profile.</p>
</section>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Developer profile</div>'
    '<div class="section-sub">Enter your details to view a salary estimate.</div>',
    unsafe_allow_html=True,
)

try:
    model, scaler_x, scaler_y = load_artifacts()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_FILE}`. Keep the model file in the same folder as app.py."
    )
    st.stop()

with st.form("salary_form"):
    st.markdown("#### Tell us about your experience")

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", list(COUNTRY_MULT))
        education = st.selectbox("Education", list(EDUCATION_MULT))
        years = st.number_input(
            "Years of professional experience",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.5,
        )
        age = st.number_input("Age", min_value=18, max_value=80, value=25, step=1)

    with col2:
        developer_type = st.selectbox("Developer type", list(DEVTYPE_MULT))
        work_history = st.selectbox("Work history", list(WORK_MULT))
        industry = st.selectbox("Industry", list(INDUSTRY_MULT))
        level = st.select_slider(
            "Developer level",
            options=list(JOB_LEVELS),
            value=3,
            format_func=lambda n: f"{n} · {JOB_LEVELS[n]}",
        )

    submitted = st.form_submit_button("✨ Predict salary", use_container_width=True)

if submitted:
    raw = predict_salary(model, scaler_x, scaler_y, years)

    monthly, level_base, exp_bonus, core, multiplier = estimate_monthly(
        raw, years, age, level, country, education,
        developer_type, work_history, industry,
    )
    annual = monthly * 12

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated annual salary</div>
        <div class="salary">₹{annual:,.0f}</div>
        <div class="result-note">
            ₹{monthly:,.0f} per month · {JOB_LEVELS[level]} level
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View salary breakdown"):
        st.write(f"Model prediction (years only): ₹{raw:,.2f} per month")
        st.write(f"Level base salary: ₹{level_base:,.2f} per month")
        st.write(f"Experience bonus ({years} yrs): ₹{exp_bonus:,.2f} per month")
        st.write(f"Subtotal before profile adjustments: ₹{core:,.2f} per month")
        st.write(
            f"Profile multiplier (country, education, type, work history, "
            f"industry, age): x{multiplier:.2f}"
        )
        st.caption(
            "The multipliers are assumptions defined at the top of app.py. "
            "Edit them to match your own data."
        )

    st.markdown("#### Salary comparison")
    st.bar_chart({
        "Monthly salary (₹)": {
            "Model prediction": max(raw, 0),
            "Level base": level_base,
            "Subtotal": core,
            "Final estimate": monthly,
        }
    })

    st.caption(
        f"Profile: {country} · {education} · {developer_type} · "
        f"{work_history} · {industry} · Age {age}"
    )
else:
    st.info("Fill in your profile and select **Predict salary** to see your estimate.")

st.markdown(
    '<div class="footnote">'
    'This is an estimate from the model included with the project, not a guaranteed salary. '
    'Actual salaries vary by location, company, skills, and market conditions.'
    '</div>',
    unsafe_allow_html=True,
)