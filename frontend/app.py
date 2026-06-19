import streamlit as st
import requests

st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide"
)

BACKEND_URL = "https://multi-agents-cdss.onrender.com"

st.title("🏥 Multi-Agent Clinical Decision Support System")

# Disease Prediction
st.header("🩺 Disease Prediction")

symptoms = st.text_input(
    "Enter Symptoms (comma separated)",
    placeholder="chills,vomiting,high_fever,sweating,headache"
)

if st.button("Analyze Patient"):

    if not symptoms.strip():
        st.warning("Please enter symptoms.")
    else:

        try:
            response = requests.get(
                f"{BACKEND_URL}/predict",
                params={"symptoms": symptoms},
                timeout=30
            )

            result = response.json()

            if "error" in result:
                st.error(result["error"])

            else:
                st.success("Analysis Complete")

                st.subheader("🦠 Predicted Disease")
                st.write(result["disease"])

                st.subheader("📖 Description")
                st.write(result["description"])

                st.subheader("📊 Severity Score")
                st.write(result["severity_score"])

                st.subheader("🚨 Risk Level")
                st.write(result["risk_level"])

                st.subheader("💊 Precautions")

                for precaution in result["precautions"]:
                    st.write("✅", precaution)

        except Exception as e:
            st.error(f"Connection Error: {e}")

# RAG Section
st.header("🤖 Ask Disease Knowledge Base")

question = st.text_input(
    "Ask a disease-related question",
    placeholder="What is malaria?"
)

if st.button("Ask RAG"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:

        try:
            response = requests.get(
                f"{BACKEND_URL}/ask_rag",
                params={"question": question},
                timeout=60
            )

            result = response.json()

            if "error" in result:
                st.error(result["error"])

            else:
                st.success("Answer Found")
                st.write(result["answer"])

        except Exception as e:
            st.error(f"Connection Error: {e}")