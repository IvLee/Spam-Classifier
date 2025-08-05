import streamlit as st
import joblib
import os

# --- Load Model & Vectorizer ---
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "../spam_classifier.pkl"))
vectorizer = joblib.load(os.path.join(base_path, "../vectorizer.pkl"))

spam_index = list(model.classes_).index("spam")
threshold = 0.4

# --- App Config ---
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📩", layout="centered")

# --- Sidebar ---
st.sidebar.title("📌 About")
st.sidebar.info(
    """
    This is a **Machine Learning Spam Detector** trained on the
    [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
    
    **Tech Stack:**
    - Python
    - scikit-learn
    - TF-IDF Vectorization
    - Naive Bayes / Logistic Regression
    - Streamlit
    """
)

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 **Author:** Your Name")
st.sidebar.write("📅 **Last Updated:** Aug 2025")

# --- Main Title ---
st.markdown("<h1 style='text-align: center;'>📩 Spam SMS Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter an SMS message below to check if it's spam or safe.</p>", unsafe_allow_html=True)

# --- Input ---
msg = st.text_area("✏️ Your Message:", height=120)

# --- Classify Button ---
if st.button("🔍 Classify Message"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        msg_tfidf = vectorizer.transform([msg])
        prob = model.predict_proba(msg_tfidf)[0][spam_index]
        label = "Spam" if prob >= threshold else "Ham"

        # --- Styled Output ---
        if label == "Spam":
            st.markdown(
                f"<div style='padding:15px; background-color:#ffcccc; border-radius:10px;'>"
                f"<h3>🚨 Prediction: Spam</h3>"
                f"<p>Spam Probability: {prob:.2%}</p></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='padding:15px; background-color:#ccffcc; border-radius:10px;'>"
                f"<h3>✅ Prediction: Ham</h3>"
                f"<p>Spam Probability: {prob:.2%}</p></div>",
                unsafe_allow_html=True
            )

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
