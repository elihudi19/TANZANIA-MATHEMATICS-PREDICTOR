"""
app.py
------
Mwanza Mathematics Performance Prediction System.

A Streamlit web app that predicts whether a Form Four student in the Mwanza
Region is likely to PASS or FAIL the NECTA Mathematics examination, using
either a Logistic Regression or a Random Forest model. The app also produces
a probability score and a personalised improvement-suggestion list built by
analysing which input variables are pulling the prediction down.

Folder expected next to this file:
    model_artifacts/
        logistic_model.pkl
        random_forest_model.pkl
        feature_config.pkl
        metrics.json
        confusion_matrix_logreg.png
        confusion_matrix_rf.png

Run locally:
    streamlit run app.py
"""

import json
import os
from datetime import datetime
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Mwanza Maths Performance Predictor",
    page_icon="📐",
    layout="wide",
)

ARTIFACT_DIR = "model_artifacts"


# --------------------------------------------------------------------------- #
# Cached loaders (so files are only read from disk once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load both trained pipelines, the feature config, and stored metrics.

    If the model_artifacts/ folder is missing or incomplete (e.g. a fresh
    Streamlit Cloud deploy where the trained .pkl files weren't committed to
    git), this automatically trains both models on the fly from
    Mwanza_Dataset.csv using train_model.py, so the app always comes up
    instead of showing a "could not load models" error.
    """
    logreg_path = os.path.join(ARTIFACT_DIR, "logistic_model.pkl")
    rf_path = os.path.join(ARTIFACT_DIR, "random_forest_model.pkl")
    config_path = os.path.join(ARTIFACT_DIR, "feature_config.pkl")
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")

    required_paths = [logreg_path, rf_path, config_path, metrics_path]

    if not all(os.path.exists(path) for path in required_paths):
        with st.spinner(
            "First-time setup: training the models from Mwanza_Dataset.csv "
            "(this only happens once)..."
        ):
            try:
                import train_model

                train_model.train_and_save()
            except Exception as exc:
                raise FileNotFoundError(
                    "Model artifacts were missing and automatic training "
                    f"failed: {exc}. Make sure 'Mwanza_Dataset.csv' is in "
                    "the same folder as app.py, or run "
                    "'python train_model.py' manually and redeploy with the "
                    "generated 'model_artifacts/' folder included."
                ) from exc

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact not found: '{path}'. "
                f"Please run 'python train_model.py' first to generate it."
            )

    logreg_model = joblib.load(logreg_path)
    rf_model = joblib.load(rf_path)
    feature_config = joblib.load(config_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return logreg_model, rf_model, feature_config, metrics


def build_input_row(school_type, ratio, attendance, has_book, mock_grade, feature_config):
    """Turn raw user selections into a single-row DataFrame matching training features."""
    school_encoded = feature_config["school_type_map"][school_type]
    book_encoded = feature_config["book_map"][has_book]
    mock_score = feature_config["mock_grade_map"][mock_grade]

    row = {
        "Teacher-to-student ratio": ratio,
        "Attendance": attendance,
        "Mock_Score": mock_score,
        "School_Type_Encoded": school_encoded,
        "Has_Book_Encoded": book_encoded,
    }
    ordered = feature_config["feature_order"]
    return pd.DataFrame([[row[col] for col in ordered]], columns=ordered)


def compute_contributions(input_row, logreg_model, feature_config):
    """
    Standardize the numeric inputs using the SAME fitted scaler used at
    training time, then multiply by the logistic regression coefficients to
    get each variable's contribution to the log-odds of passing.
    A negative contribution means that variable is currently pushing the
    student toward failing; a positive (or zero) contribution means that
    variable is at, or pushing toward, its best possible state.

    For scaled numeric features (Teacher-to-student ratio, Attendance,
    Mock_Score) the contribution is simply scaled_value * coefficient, since
    0 on the standardized scale represents the dataset average.

    For categorical 0/1 encoded features (School_Type_Encoded,
    Has_Book_Encoded) the raw encoded value has no such "neutral" meaning —
    0 and 1 are just labels, not distances from an average. Categorical
    features are scored relative to their best-outcome category (best = 1
    if the coefficient is positive, else 0), so being in the worse category
    always correctly shows up as a negative contribution.
    """
    preprocessor = logreg_model.named_steps["preprocessor"]
    transformed = preprocessor.transform(input_row)  # numeric(scaled) + passthrough
    coefficients = feature_config["logreg_coefficients"]
    feature_order = feature_config["feature_order"]
    categorical_features = set(
        feature_config.get("categorical_encoded_features", ["School_Type_Encoded", "Has_Book_Encoded"])
    )

    contributions = {}
    for i, feat in enumerate(feature_order):
        value = float(transformed[0][i])
        coef = float(coefficients[feat])

        if feat in categorical_features:
            best_value = 1.0 if coef > 0 else 0.0
            contributions[feat] = (value - best_value) * coef
        else:
            contributions[feat] = value * coef

    return contributions


FRIENDLY_NAMES = {
    "Teacher-to-student ratio": "Teacher-to-student ratio",
    "Attendance": "Attendance rate",
    "Mock_Score": "Mock examination grade",
    "School_Type_Encoded": "School type",
    "Has_Book_Encoded": "Mathematics book ownership",
}


def suggestion_for(feature, value_raw, lang="English"):
    """Return an actionable suggestion string for a risk factor, in the chosen language."""
    suggestions_en = {
        "Teacher-to-student ratio": (
            f"The teacher-to-student ratio ({value_raw}:1) is high. Advocate for smaller "
            "class sizes, extra tutoring sessions, or peer study groups to compensate for "
            "reduced individual attention."
        ),
        "Attendance": (
            f"Attendance is at {value_raw}%. Improving regular class attendance is one of "
            "the strongest levers for exam performance — aim to close any attendance gaps."
        ),
        "Mock_Score": (
            "The mock examination result suggests weak readiness. Focus revision on the "
            "topics missed in the mock exam and consider a structured past-papers practice plan."
        ),
        "School_Type_Encoded": (
            "The school type is associated with historically lower NECTA pass rates in this "
            "dataset. Extra resource support (learning materials, tutoring) can help offset this."
        ),
        "Has_Book_Encoded": (
            "The student does not currently own a Mathematics textbook. Providing access to a "
            "textbook (personal, borrowed, or library copy) is strongly associated with better outcomes."
        ),
    }
    suggestions_sw = {
        "Teacher-to-student ratio": (
            f"Uwiano wa mwalimu kwa wanafunzi ({value_raw}:1) ni mkubwa. Shauri madarasa "
            "madogo, masomo ya ziada, au vikundi vya kujifunza pamoja ili kuziba pengo la "
            "uangalizi wa mtu mmoja mmoja."
        ),
        "Attendance": (
            f"Mahudhurio yako kwa asilimia {value_raw}%. Kuboresha mahudhurio darasani mara "
            "kwa mara ni njia kuu ya kuboresha matokeo ya mtihani."
        ),
        "Mock_Score": (
            "Matokeo ya mtihani wa mtihani wa maandalizi (mock) yanaonyesha maandalizi hafifu. "
            "Weka mkazo kwenye mada zilizoshindwa kwenye mock na fanya mazoezi ya maswali ya nyuma."
        ),
        "School_Type_Encoded": (
            "Aina ya shule inahusiana na kiwango cha chini cha ufaulu wa NECTA kwenye data hii. "
            "Msaada wa ziada wa vifaa vya kujifunzia na ufundishaji unaweza kusaidia."
        ),
        "Has_Book_Encoded": (
            "Mwanafunzi hana kitabu cha Hisabati kwa sasa. Kupata kitabu (binafsi, mkopo, au "
            "cha maktaba) kunahusiana sana na matokeo bora."
        ),
    }
    table = suggestions_en if lang == "English" else suggestions_sw
    return table.get(feature, "Review this factor with a teacher for tailored advice.")


def strength_note_for(feature, value_raw, lang="English"):
    """Positive reinforcement message for factors already working in the student's favor."""
    notes_en = {
        "Teacher-to-student ratio": f"A teacher-to-student ratio of {value_raw}:1 is favorable — keep it up.",
        "Attendance": f"Attendance at {value_raw}% is strong and supporting the prediction well.",
        "Mock_Score": "The mock examination grade is a strong positive signal — maintain this momentum.",
        "School_Type_Encoded": "The school type is currently working in the student's favor.",
        "Has_Book_Encoded": "Owning a Mathematics textbook is helping this student's chances — make sure it's being put to regular use.",
    }
    notes_sw = {
        "Teacher-to-student ratio": f"Uwiano wa {value_raw}:1 ni mzuri — endelea hivyo.",
        "Attendance": f"Mahudhurio ya {value_raw}% ni mazuri na yanasaidia matokeo.",
        "Mock_Score": "Alama ya mock ni ishara nzuri — endelea na kasi hiyo.",
        "School_Type_Encoded": "Aina ya shule kwa sasa inamsaidia mwanafunzi huyu.",
        "Has_Book_Encoded": "Kumiliki kitabu cha Hisabati kunamsaidia mwanafunzi huyu — hakikisha kinatumika mara kwa mara.",
    }
    table = notes_en if lang == "English" else notes_sw
    return table.get(feature, "This factor is currently helping.")


def generate_pdf_report(
    school_type, ratio, attendance, has_book, mock_grade, model_choice,
    prediction, probability_pass, contributions, raw_values, language
):
    """Generate a PDF report of the prediction and suggestions."""
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        spaceBefore=10
    )

    report_title = "Mwanza Mathematics Performance Prediction Report" if language == "English" else "Ripoti ya Utabiri wa Utendaji wa Hisabati wa Mwanza"
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.2*inch))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    info_title = "Student Information" if language == "English" else "Habari za Mwanafunzi"
    story.append(Paragraph(info_title, heading_style))

    student_info = [
        ["School Type", school_type],
        ["Teacher-to-student Ratio", f"{ratio}:1"],
        ["Attendance Rate", f"{attendance}%"],
        ["Mathematics Book", has_book],
        ["Mock Examination Grade", mock_grade],
    ]
    if language == "Swahili":
        student_info = [
            ["Aina ya Shule", school_type],
            ["Uwiano wa Mwalimu kwa Wanafunzi", f"{ratio}:1"],
            ["Kiwango cha Mahudhurio", f"{attendance}%"],
            ["Kitabu cha Hisabati", has_book],
            ["Alama ya Mtihani wa Maandalizi", mock_grade],
        ]

    t = Table(student_info, colWidths=[3*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    result_title = "Prediction Result" if language == "English" else "Matokeo ya Utabiri"
    story.append(Paragraph(result_title, heading_style))

    result_text = "PASS ✅" if prediction == 1 else "FAIL ❌"
    prob_text = f"{probability_pass * 100:.1f}%"
    model_text = f"Model: {model_choice}"

    prediction_info = [
        ["Predicted Outcome", result_text],
        ["Probability of Passing", prob_text],
        ["Model Used", model_text],
    ]
    if language == "Swahili":
        prediction_info = [
            ["Matokeo ya Kutabiri", result_text],
            ["Uwezekano wa Kupitisha", prob_text],
            ["Mtindo Uliotumika", model_text],
        ]

    t2 = Table(prediction_info, colWidths=[3*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*inch))

    sugg_title = "Personalised Suggestions" if language == "English" else "Ushauri wa Kibinafsi"
    story.append(Paragraph(sugg_title, heading_style))

    sorted_factors = sorted(contributions.items(), key=lambda kv: kv[1])
    risk_factors = [f for f in sorted_factors if f[1] < 0]

    if not risk_factors:
        msg = (
            "All measured factors are currently working in this student's favor. "
            "Keep up the strong routine!"
            if language == "English"
            else "Vigezo vyote vinavyopimwa kwa sasa vinamsaidia mwanafunzi huyu. "
            "Endelea na mwenendo huu mzuri!"
        )
        story.append(Paragraph(msg, styles['Normal']))
    else:
        focus_msg = (
            "All factors below are currently reducing this student's chance of passing, "
            "ranked from most to least impactful:"
            if language == "English"
            else "Vigezo vyote hapa chini kwa sasa vinapunguza nafasi ya mwanafunzi kufaulu, "
            "kwa mpangilio wa athari kubwa hadi ndogo:"
        )
        story.append(Paragraph(focus_msg, styles['Normal']))
        story.append(Spacer(1, 0.15*inch))

        for i, (feat, contrib) in enumerate(risk_factors, start=1):
            friendly = FRIENDLY_NAMES.get(feat, feat)
            text = suggestion_for(feat, raw_values[feat], lang=language)
            story.append(Paragraph(f"<b>{i}. {friendly}</b>", styles['Normal']))
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

    # Factors Already Working Well Section
    positive_factors = [f for f in sorted_factors if f[1] >= 0]
    if positive_factors:
        story.append(Spacer(1, 0.15*inch))
        strengths_title = "✅ Factors Already Working Well" if language == "English" else "✅ Mambo Yanayosaidia Tayari"
        story.append(Paragraph(strengths_title, heading_style))

        for feat, contrib in positive_factors:
            friendly = FRIENDLY_NAMES.get(feat, feat)
            note = strength_note_for(feat, raw_values[feat], lang=language)
            story.append(Paragraph(f"<b>{friendly}</b> — {note}", styles['Normal']))
            story.append(Spacer(1, 0.08*inch))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


# --------------------------------------------------------------------------- #
# Load everything
# --------------------------------------------------------------------------- #
try:
    logreg_model, rf_model, feature_config, stored_metrics = load_artifacts()
    load_error = None
except Exception as exc:  # noqa: BLE001
    logreg_model = rf_model = feature_config = stored_metrics = None
    load_error = str(exc)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("📐 Mwanza Maths Predictor")
st.sidebar.caption(
    "Predicting Form Four NECTA Mathematics outcomes for schools in the "
    "Mwanza Region, Tanzania."
)
language = st.sidebar.radio("Suggestion language / Lugha", ["English", "Swahili"], index=0)
model_choice = st.sidebar.radio(
    "Prediction model", ["Logistic Regression", "Random Forest"], index=0
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built for the EASTC Bachelor of Data Science capstone: "
    "*Mwanza Mathematics Performance Prediction System*."
)

# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
st.title("Mwanza Mathematics Performance Prediction System")
st.write(
    "Enter a student's details below to predict whether they are likely to "
    "**PASS** or **FAIL** the NECTA Mathematics examination, see the "
    "probability score, and get personalised improvement suggestions."
)

if load_error:
    st.error(
        "Could not load the trained models. Please make sure you have run "
        "`python train_model.py` in this project folder first, and that the "
        f"`model_artifacts/` directory sits next to `app.py`.\n\nDetails: {load_error}"
    )
    st.stop()

tab_predict, tab_performance = st.tabs(["🔮 Prediction", "📊 Model Performance"])

# --------------------------------------------------------------------------- #
# Prediction tab
# --------------------------------------------------------------------------- #
with tab_predict:
    with st.form(key="metrics_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Parameters")
            school_type = st.selectbox("School Type", ["Private", "Government"])

            ratio = st.number_input(
                label="Teacher-to-student ratio (students per teacher)",
                min_value=50,
                max_value=300,
                value=100,
                step=1,
            )

            attendance = st.number_input(
                label="Attendance rate (%)",
                min_value=50,
                max_value=99,
                value=80,
                step=1,
            )

        with col2:
            has_book = st.selectbox("Does the student own a Mathematics book?", ["Own a Book", "Not Own Book"])
            mock_grade = st.selectbox("Mock examination grade", ["A", "B", "C", "D", "F"])

            st.markdown("")
            predict_clicked = st.form_submit_button("Predict Result", type="primary")

    if predict_clicked:
        try:
            input_row = build_input_row(
                school_type, ratio, attendance, has_book, mock_grade, feature_config
            )

            chosen_model = logreg_model if model_choice == "Logistic Regression" else rf_model
            prediction = int(chosen_model.predict(input_row)[0])
            probability_pass = float(chosen_model.predict_proba(input_row)[0][1])

            st.markdown("---")
            result_col, prob_col = st.columns([1, 1])

            with result_col:
                if prediction == 1:
                    st.success("### ✅ Predicted Result: PASS")
                else:
                    st.error("### ❌ Predicted Result: FAIL")
                st.caption(f"Model used: {model_choice}")

            with prob_col:
                st.metric("Probability of Passing", f"{probability_pass * 100:.1f}%")
                st.progress(min(max(probability_pass, 0.0), 1.0))

            st.markdown("---")
            st.subheader(
                "💡 Personalised Suggestions" if language == "English" else "💡 Ushauri wa Kibinafsi"
            )

            contributions = compute_contributions(input_row, logreg_model, feature_config)
            raw_values = {
                "Teacher-to-student ratio": ratio,
                "Attendance": attendance,
                "Mock_Score": mock_grade,
                "School_Type_Encoded": school_type,
                "Has_Book_Encoded": has_book,
            }

            sorted_factors = sorted(contributions.items(), key=lambda kv: kv[1])

            risk_factors = [f for f in sorted_factors if f[1] < 0]
            positive_factors = [f for f in sorted_factors if f[1] >= 0]

            if not risk_factors:
                msg = (
                    "All measured factors are currently working in this student's favor. "
                    "Keep up the strong routine!"
                    if language == "English"
                    else "Vigezo vyote vinavyopimwa kwa sasa vinamsaidia mwanafunzi huyu. "
                    "Endelea na mwenendo huu mzuri!"
                )
                st.info(msg)
            else:
                st.write(
                    "All factors below are currently reducing this student's chance of "
                    "passing, ranked from most to least impactful:"
                    if language == "English"
                    else "Vigezo vyote hapa chini kwa sasa vinapunguza nafasi ya mwanafunzi "
                    "kufaulu, kwa mpangilio wa athari kubwa hadi ndogo:"
                )
                for i, (feat, contrib) in enumerate(risk_factors, start=1):
                    friendly = FRIENDLY_NAMES.get(feat, feat)
                    text = suggestion_for(feat, raw_values[feat], lang=language)
                    st.markdown(f"**{i}. {friendly}** — {text}")

            if positive_factors:
                with st.expander(
                    "✅ Factors already working well" if language == "English" else "✅ Mambo yanayosaidia tayari"
                ):
                    for feat, contrib in positive_factors:
                        friendly = FRIENDLY_NAMES.get(feat, feat)
                        note = strength_note_for(feat, raw_values[feat], lang=language)
                        st.markdown(f"- **{friendly}** — {note}")

            st.markdown("---")
            pdf_buffer = generate_pdf_report(
                school_type, ratio, attendance, has_book, mock_grade, model_choice,
                prediction, probability_pass, contributions, raw_values, language
            )

            download_label = "📥 Download Student Report (PDF)" if language == "English" else "📥 Pakua Ripoti ya Mwanafunzi (PDF)"
            st.download_button(
                label=download_label,
                data=pdf_buffer,
                file_name=f"student_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while generating the prediction: {exc}")

# --------------------------------------------------------------------------- #
# Model performance tab
# --------------------------------------------------------------------------- #
with tab_performance:
    st.subheader("Model Evaluation Metrics")
    st.caption(
        f"Evaluated on a held-out test set of {stored_metrics.get('test_size', 'N/A')} "
        f"students (out of {stored_metrics.get('dataset_size', 'N/A')} total records)."
    )

    metric_col1, metric_col2 = st.columns(2)

    for col, key, title, cm_file in [
        (metric_col1, "logistic_regression", "Logistic Regression", "confusion_matrix_logreg.png"),
        (metric_col2, "random_forest", "Random Forest", "confusion_matrix_rf.png"),
    ]:
        with col:
            st.markdown(f"#### {title}")
            m = stored_metrics.get(key, {})
            st.write(f"**Accuracy:** {m.get('accuracy', 'N/A') * 100:.2f}%" if m.get("accuracy") is not None else "N/A")
            st.write(f"**Precision:** {m.get('precision', 'N/A')}")
            st.write(f"**Recall:** {m.get('recall', 'N/A')}")
            st.write(f"**F1 Score:** {m.get('f1_score', 'N/A')}")

            cm_path = os.path.join(ARTIFACT_DIR, cm_file)
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"{title} Confusion Matrix", use_container_width=True)
            else:
                st.warning(f"Confusion matrix image not found: {cm_path}")

    st.markdown("---")
    st.caption(
        "Note: the suggestion engine always uses the Logistic Regression coefficients "
        "to explain factor contributions, since they are directly interpretable as "
        "log-odds — even when Random Forest is selected for the headline prediction."
    )
