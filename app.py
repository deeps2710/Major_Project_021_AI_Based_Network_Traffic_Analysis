
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_PATH = Path("models/traffic_model.joblib")
METRICS_PATH = Path("reports/metrics.json")

st.set_page_config(
    page_title="AI Network Traffic Analysis",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI-Based Network Traffic Analysis")
st.caption(
    "Machine-learning-based network intrusion detection "
    "using labeled flow data."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. Run "
        "`python train_model.py` from the project root first."
    )
    st.stop()

# Load the saved model and its required feature schema.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Analyze Traffic", "About"],
)

if page == "Dashboard":
    st.subheader("Model performance")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric(
            "Attack precision",
            f"{metrics['precision_attack']:.2%}",
        )
        c3.metric(
            "Attack recall",
            f"{metrics['recall_attack']:.2%}",
        )
        c4.metric("Attack F1-score", f"{metrics['f1_attack']:.2%}")

        cm = np.array(metrics["confusion_matrix"])

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted benign", "Predicted attack"],
            y=["Actual benign", "Actual attack"],
            labels={
                "x": "Prediction",
                "y": "Actual class",
                "color": "Number of flows",
            },
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "These metrics come from the held-out test split "
            "used during training, not from the currently uploaded file."
        )
    else:
        st.warning(
            "Evaluation report not found. Run the training script first."
        )

    st.subheader("Model information")
    st.write(f"Number of input features: {len(features)}")
    st.write("Classes: BENIGN and ATTACK")

elif page == "Analyze Traffic":
    st.subheader("Analyze a network-flow CSV")

    st.write(
        "Upload a CSV containing the same kind of numeric flow "
        "features used during training. Extra columns are ignored."
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            df.columns = (
                df.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )

            if df.empty:
                st.error("The uploaded CSV contains no rows.")
                st.stop()

            st.write("Input preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Build the exact feature schema expected by the model.
            # Missing features become NaN and are imputed by the
            # fitted preprocessing pipeline.
            X = pd.DataFrame(index=df.index)

            for col in features:
                if col in df.columns:
                    X[col] = pd.to_numeric(
                        df[col], errors="coerce"
                    )
                else:
                    X[col] = np.nan

            X = X.replace([np.inf, -np.inf], np.nan)

            missing_count = sum(
                col not in df.columns for col in features
            )

            if missing_count:
                st.warning(
                    f"{missing_count} expected feature columns "
                    "were missing. They will be imputed, which may "
                    "reduce prediction reliability."
                )

            if X.isna().all(axis=1).any():
                st.warning(
                    "Some rows have no usable values for the model's "
                    "features. Predictions for these rows may be unreliable."
                )

            if st.button("Analyze traffic", type="primary"):
                with st.spinner("Analyzing network flows..."):
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)

                results = df.copy()
                results["Prediction"] = [
                    "ATTACK" if value == 1 else "BENIGN"
                    for value in predictions
                ]
                results["Attack Probability (%)"] = (
                    probabilities[:, 1] * 100
                ).round(2)

                total = len(results)
                attack_count = int((predictions == 1).sum())
                benign_count = total - attack_count
                attack_percentage = 100 * attack_count / total

                st.subheader("Analysis results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total flows", f"{total:,}")
                c2.metric("Predicted attacks", f"{attack_count:,}")
                c3.metric("Predicted benign", f"{benign_count:,}")

                st.metric(
                    "Predicted attack percentage",
                    f"{attack_percentage:.2f}%",
                )

                counts = pd.DataFrame({
                    "Class": ["BENIGN", "ATTACK"],
                    "Flows": [benign_count, attack_count],
                })

                fig = px.bar(
                    counts,
                    x="Class",
                    y="Flows",
                    color="Class",
                    title="Predicted Traffic Distribution",
                    text="Flows",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Filter predictions")

                selected_class = st.selectbox(
                    "Show traffic type",
                    ["ALL", "ATTACK", "BENIGN"],
                )

                filtered = results
                if selected_class != "ALL":
                    filtered = results[
                        results["Prediction"] == selected_class
                    ]

                st.dataframe(
                    filtered,
                    use_container_width=True,
                    height=400,
                )

                st.download_button(
                    "Download analysis results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="traffic_analysis_results.csv",
                    mime="text/csv",
                )

                st.warning(
                    "Predictions are model estimates, not proof that "
                    "a connection is malicious. Validate alerts before "
                    "taking security action."
                )

        except Exception as exc:
            st.error(f"Could not analyze the uploaded file: {exc}")

elif page == "About":
    st.subheader("About this project")

    st.markdown("""
    **Objective:** Identify potentially malicious network flows
    using supervised machine learning.

    **Model:** Random Forest classifier.

    **Input:** CSV records containing numeric network-flow features.

    **Output:** Predicted benign/attack class and attack probability.

    **Dataset:** CICIDS2017, with its original dataset citation
    included in the project report.
    """)

    st.info(
        "This prototype analyzes uploaded records. It does not "
        "capture live packets or automatically block network traffic."
    )
