import streamlit as st
import requests

st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Multi-Agent Clinical Decision Support System")

BACKEND_URL = "https://multi-agents-cdss.onrender.com"

symptoms = st.text_input(
    "Enter Symptoms (comma separated)"
)

if st.button("Analyze Patient"):

    response = requests.get(
        f"{BACKEND_URL}/predict",
        params={"symptoms": symptoms}
    )

    st.write("Status Code:", response.status_code)
    st.write("Response Text:", response.text)

    try:
        result = response.json()

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
        st.error(f"Error: {e}")

st.subheader("🤖 Ask About Disease (RAG)")

question = st.text_input(
    "Ask a question"
)

if st.button("Ask RAG"):

    response = requests.get(
        f"{BACKEND_URL}/ask_rag",
        params={"question": question}
    )

    st.write("Status Code:", response.status_code)
    st.write("Response Text:", response.text)

    try:
        result = response.json()

        st.success("RAG Answer")
        st.write(result["answer"])

    except Exception as e:
        st.error(f"Error: {e}")