"""
Spam SMS Classifier Web App
---------------------------
- Loads saved model & vectorizer
- Lets user enter a message in a browser
- Displays prediction & probability
"""

import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Find spam index for probability
spam_index = list(model.classes_).index("spam")
threshold = 0.3  # Same as training

# Streamlit UI
st.title("📩 Spam SMS Classifier")
st.write("Enter a message below to check if it's spam or ham.")

msg = st.text_area("Your Message:")

if st.button("Classify"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        msg_tfidf = vectorizer.transform([msg])
        prob = model.predict_proba(msg_tfidf)[0][spam_index]
        label = "Spam" if prob >= threshold else "Ham"

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Spam Probability: {prob:.2%}")

        if label == "Spam":
            st.error("⚠️ This message looks suspicious!")
        else:
            st.success("✅ This message looks safe.")
