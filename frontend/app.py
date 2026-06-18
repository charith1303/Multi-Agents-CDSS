import streamlit as st
import requests

st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Multi-Agent Clinical Decision Support System")

symptoms = st.text_input(
    "Enter Symptoms (comma separated)"
)

if st.button("Analyze Patient"):

    response = requests.get(
    "https://multi-agents-cdss.onrender.com/predict",
    params={"symptoms": symptoms}
    )

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
        st.divider()

st.subheader("🤖 Ask About Disease (RAG)")

question = st.text_input(
    "Ask a question"
)

if st.button("Ask RAG"):

    response = requests.get(
    "https://multi-agents-cdss.onrender.com/ask_rag",
    params={"question": question}

    )

    result = response.json()

    st.success("RAG Answer")

    st.write(result["answer"])