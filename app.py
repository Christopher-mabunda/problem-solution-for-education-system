"""EduPrediction - Student Risk Prediction System (Streamlit app).

Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src import model as M
from src.config import (DATA_FILE, FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                        RISK_COLOURS, RISK_LABELS, grade_to_risk)
from src.data_loader import clean, load_raw
from src.recommender import RISK_SUMMARY, chatbot_reply, recommend

st.set_page_config(page_title="EduPrediction", page_icon="🎓", layout="wide")

NAVY, BLUE = "#0b2150", "#2f6fde"
st.markdown(
    f"""
    <style>
      .hero {{background:{NAVY}; padding:28px 32px; border-radius:10px; color:white; margin-bottom:18px;}}
      .hero small {{letter-spacing:3px; color:#a9bce6; font-weight:600;}}
      .hero h1 {{color:white; margin:4px 0 0 0; font-size:2.6rem;}}
      .hero p {{color:#cfd9f2; margin:4px 0 0 0;}}
      .risk-card {{padding:22px; border-radius:10px; color:white; text-align:center;}}
      .risk-card h2 {{color:white; margin:0; font-size:2.2rem;}}
      .risk-card p {{margin:4px 0 0 0; color:white;}}
    </style>
    <div class="hero">
      <small>AI PROJECT | EDUCATION ANALYTICS</small>
      <h1>EduPrediction</h1>
      <p>Student Risk Prediction System &mdash; identify at-risk students early and guide them to the right support.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return M.load()


try:
    bundle, meta = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Open a terminal in the project folder and run `python train_model.py`, then refresh this page.")
    st.stop()

defaults = meta["defaults"]

st.sidebar.title("EduPrediction")
page = st.sidebar.radio("Navigate", ["Assessment", "Dashboard", "Model Performance", "Chatbot", "About"])
model_choice = st.sidebar.selectbox(
    "Model", list(bundle["models"].keys()),
    index=list(bundle["models"].keys()).index(bundle["best"]),
    help="Best performing model is selected by default.",
)
st.sidebar.caption("ANSWER. ANALYSE. ACT.")


# ---------------------------------------------------------------------------
def risk_card(risk: str, probs: dict):
    st.markdown(
        f'<div class="risk-card" style="background:{RISK_COLOURS[risk]}">'
        f"<p>PREDICTED RISK LEVEL</p><h2>{risk}</h2><p>{RISK_SUMMARY[risk]}</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption("Model confidence")
    for lbl in RISK_LABELS:
        st.progress(probs[lbl], text=f"{lbl}: {probs[lbl]*100:.0f}%")


# ---------------------------------------------------------------------------
# 1. Assessment
# ---------------------------------------------------------------------------
if page == "Assessment":
    st.subheader("Student risk assessment")
    st.write("Answer five short questions to get a risk level and recommended actions.")
  
 # Create two columns for the assessment and results.
    left, right = st.columns([1.1, 1])
  
    with left:
        with st.form("assessment"):
            c1, c2 = st.columns(2)

 # Collect basic student information.
            age = c1.number_input("1. Age", 14, 25, 17)
            sex = c2.selectbox("2. Gender", ["F", "M"], format_func=lambda x: "Female" if x == "F" else "Male")
 # Collect study time and attendance information.
            studytime = st.select_slider(
                "3. Weekly study time", options=[1, 2, 3, 4], value=2,
                format_func=lambda x: {1: "< 2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: "> 10 hrs"}[x])
            absences = st.slider("4. Number of absences", 0, 93, 3)
            g2 = st.slider("5. Second period grade (G2, 0-20)", 0, 20, 11)

 # Collect optional academic and family details.          
            with st.expander("Optional: more details (improves accuracy)"):
                a1, a2 = st.columns(2)
                failures = a1.number_input("Previous class failures", 0, 4, int(defaults["failures"]))
                g1 = a2.number_input("First period grade (G1)", 0, 20, g2)
                medu = a1.slider("Mother's education (0-4)", 0, 4, int(defaults["Medu"]))
                fedu = a2.slider("Father's education (0-4)", 0, 4, int(defaults["Fedu"]))
                famsup = a1.selectbox("Family educational support", ["yes", "no"],
                                      index=["yes", "no"].index(defaults["famsup"]))
                schoolsup = a2.selectbox("Extra school support", ["yes", "no"],
                                         index=["yes", "no"].index(defaults["schoolsup"]))
                internet = a1.selectbox("Internet access at home", ["yes", "no"],
                                        index=["yes", "no"].index(defaults["internet"]))
            submitted = st.form_submit_button("Analyse", type="primary", width="stretch")

 # Analyse the student's information when submitted.
    if submitted:
        record = dict(age=age, sex=sex, studytime=studytime, absences=absences,
                      failures=failures, Medu=medu, Fedu=fedu, G1=g1, G2=g2,
                      famsup=famsup, schoolsup=schoolsup, internet=internet)
      
 # Predict the student's risk level and generate recommendations.      
        risk, probs = M.predict_one(bundle, record, model_choice)
        tips = recommend(record, risk)
        st.session_state["last_result"] = {"risk": risk, "probs": probs, "tips": tips, "record": record}

    result = st.session_state.get("last_result")
    with right:
        if result:
# Display the predicted risk and recommendations.
            risk_card(result["risk"], result["probs"])
            st.markdown("#### Recommendations")
            for t in result["tips"]:
                st.markdown(f"- {t}")

 # Remind users that predictions are guidance.         
            st.caption("Predictions are guidance, not a final judgement. Combine with professional insight.")
        else: 
# Display a message before an assessment is completed.      
            st.info("Fill in the form and press **Analyse** to see the result here.")

# ---------------------------------------------------------------------------
# 2. Dashboard
# ---------------------------------------------------------------------------
elif page == "Dashboard":
    st.subheader("Lecturer / administrator dashboard")

# Allow users to choose where student data comes from.
    src = st.radio("Data source", ["Training dataset (student-por)", "Upload a CSV"], horizontal=True)

    df = None

  # Load the existing training dataset.
    if src.startswith("Training"):
        if DATA_FILE.exists():
            df = clean(load_raw())
        else:
            st.warning("Training dataset not found in data/.")
    else:
        st.caption("Required columns: " + ", ".join(FEATURES) + ". Optional: G3, plus any ID/name column.")
        up = st.file_uploader("CSV file (comma or semicolon separated)", type="csv")
        if up is not None:
            raw = pd.read_csv(up, sep=None, engine="python")

  # Check for missing required columns.
            missing = [c for c in FEATURES if c not in raw.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df = raw.copy()
                for c in NUMERIC_FEATURES:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(defaults[c])
                for c in CATEGORICAL_FEATURES:
                    df[c] = df[c].fillna(defaults[c]).astype(str)

 # Analyse the student data when available.  
    if df is not None and len(df):
        scored = M.predict_many(bundle, df, model_choice)
        counts = scored["predicted_risk"].value_counts().reindex(RISK_LABELS).fillna(0).astype(int)

 # Display student totals for each risk category.     
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Students", len(scored))
        k2.metric("High risk", counts["High"])
        k3.metric("Medium risk", counts["Medium"])
        k4.metric("Low risk", counts["Low"])

        c1, c2 = st.columns(2)
        with c1:
          
  # Create a chart showing student risk levels.        
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.bar(RISK_LABELS, counts.values, color=[RISK_COLOURS[r] for r in RISK_LABELS])
            ax.set_title("Predicted risk distribution"); ax.set_ylabel("Students")
            st.pyplot(fig)
        with c2:

# Create a chart comparing absences across risk levels.          
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            sns.boxplot(data=scored, x="predicted_risk", y="absences", order=RISK_LABELS,
                        palette=[RISK_COLOURS[r] for r in RISK_LABELS], hue="predicted_risk",
                        legend=False, ax=ax)
            ax.set_title("Absences by predicted risk")
            st.pyplot(fig)

        st.markdown("#### Student list")

# Allow users to filter students by risk level
        pick = st.multiselect("Filter by risk level", RISK_LABELS, default=["High"])
        view = scored[scored["predicted_risk"].isin(pick)] if pick else scored
        st.dataframe(view, width="stretch", height=320)
        st.download_button("Download predictions (CSV)", scored.to_csv(index=False).encode(),
                           "eduprediction_results.csv", "text/csv")

# ---------------------------------------------------------------------------
# 3. Model performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model performance")

  # Display the number of students used for trainin
    st.write(f"Trained on **{meta['n_rows']}** students. Best model: **{meta['best_model']}**.")
    table = pd.DataFrame({
        n: {"Test accuracy": r["test_accuracy"], "Test macro-F1": r["test_f1_macro"],
            "CV macro-F1": r["cv_f1_macro"]} for n, r in meta["results"].items()
    }).T
    st.dataframe(table, width="stretch")

    c1, c2 = st.columns(2)
    for col, (name, r) in zip((c1, c2), meta["results"].items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.4))
            sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=RISK_LABELS, yticklabels=RISK_LABELS, ax=ax)

          
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)

 # Display the confusion matrix on the dashboard.          
            st.pyplot(fig)

    st.markdown("#### Feature importance (best model)")
    top = meta["importances"][:10]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x=[v for _, v in top], y=[k for k, _ in top], color=BLUE, ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# 4. Chatbot
# ---------------------------------------------------------------------------
elif page == "Chatbot":
    st.subheader("Recommendation chatbot")
    st.caption("Ask about risk levels, study time, attendance or what to do next. "
               "Run an assessment first for personalised advice.")

  # Check whether a chat conversation already exists.  
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "Hi! Ask me about risk levels or say 'what should I do next?'")]

   # Display all existing messages in the conversation.
    for role, text in st.session_state.chat:
        st.chat_message(role).markdown(text)

  # Allow the user to type a question into the chatbot.
    if prompt := st.chat_input("Type your question..."):
        st.session_state.chat.append(("user", prompt))

   # Generate a response using the chatbot function.     
        reply = chatbot_reply(prompt, st.session_state.get("last_result"))
        st.session_state.chat.append(("assistant", reply))
        st.rerun()

# ---------------------------------------------------------------------------
# 5. About
# ---------------------------------------------------------------------------
else:
  # Display information about the system using Markdown.
    st.subheader("About EduPrediction")
    st.markdown(
        """
**EduPrediction** is an AI-based early warning system that predicts whether a student is at
**Low, Medium or High** academic risk and recommends what to do next.

| Risk | Final grade (G3) | Meaning |
|---|---|---|
| Low | 14 or higher | On track |
| Medium | 10 to 13 | Monitor closely |
| High | below 10 | Needs support |

**How it works:** Start, Answer five short questions, Analyse (Decision Tree and Random Forest), Review risk level and advice.

**Intended users:** teachers and counsellors, school administrators, students and parents.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit.
        """
    )
