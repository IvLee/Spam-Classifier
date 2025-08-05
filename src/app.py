import streamlit as st
import joblib
import os
import pandas as pd

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
    **Spam SMS Classifier**
    
    Detects whether a message is spam or ham.
    
    **Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
    
    **Tech Stack:**
    - Python
    - scikit-learn
    - TF-IDF Vectorization
    - Naive Bayes / Logistic Regression
    - Streamlit
    """
)

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 **Author:** Ivan Lee")
st.sidebar.write("📅 **Last Updated:** Aug 2025")

# --- Main Title ---
st.markdown(
    "<h1 style='text-align: center;'>📩 Spam SMS Classifier</h1>"
    "<p style='text-align: center;'>Classify single or multiple messages as spam or safe.</p>",
    unsafe_allow_html=True
)

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Single Message", "📂 Batch Upload"])

# --- Tab 1: Single Message ---
with tab1:
    msg = st.text_area("✏️ Your Message:", height=120)

    if st.button("Classify Single Message", key="single"):
        if msg.strip() == "":
            st.warning("⚠️ Please enter a message first.")
        else:
            msg_tfidf = vectorizer.transform([msg])
            prob = model.predict_proba(msg_tfidf)[0][spam_index]
            label = "Spam" if prob >= threshold else "Ham"

            box_style = """
                padding:15px; 
                border-radius:10px; 
                border: 2px solid;
                margin-top:10px;
            """

            if label == "Spam":
                st.markdown(
                    f"<div style='{box_style} border-color:#ff4d4d;'>"
                    f"<h3>🚨 Prediction: Spam</h3>"
                    f"<p>Spam Probability: {prob:.2%}</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='{box_style} border-color:#4dff88;'>"
                    f"<h3>✅ Prediction: Ham</h3>"
                    f"<p>Spam Probability: {prob:.2%}</p></div>",
                    unsafe_allow_html=True
                )

# --- Tab 2: Batch Upload ---
with tab2:
    uploaded_file = st.file_uploader("Upload a CSV, TXT, or Excel file", type=["csv", "txt", "xlsx"])

    if uploaded_file is not None:
        try:
            # --- Try CSV ---
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if df.shape[1] == 1:  
                    df.columns = ["message"]
                elif "message" not in df.columns:
                    st.error("CSV must have a 'message' column.")
                    st.stop()

            # --- Try Excel ---
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                if df.shape[1] == 1:  
                    df.columns = ["message"]
                elif "message" not in df.columns:
                    st.error("Excel file must have a 'message' column.")
                    st.stop()

            # --- Try TXT ---
            elif uploaded_file.name.endswith(".txt"):
                messages = uploaded_file.read().decode("utf-8").splitlines()
                df = pd.DataFrame(messages, columns=["message"])

            else:
                st.error("Unsupported file format.")
                st.stop()

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # Show file stats
        st.write(f"📄 Loaded {len(df)} messages")

        # Transform and predict
        msg_tfidf = vectorizer.transform(df["message"])
        probs = model.predict_proba(msg_tfidf)[:, spam_index]
        df["prediction"] = ["Spam" if p >= threshold else "Ham" for p in probs]
        df["spam_probability"] = [f"{p:.2%}" for p in probs]

        # Show predictions table
        st.dataframe(df)

        # Create output filename based on uploaded file
        base_filename = uploaded_file.name.rsplit(".", 1)[0]  # Remove extension
        output_filename = f"{base_filename}_predictions.csv"

        # Download predictions
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button(
        label=f"📥 Download Predictions ({output_filename})",
        data=csv_download,
        file_name=output_filename,
        mime="text/csv"
)
