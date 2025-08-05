"""
Predict messages using saved Spam SMS Classifier
------------------------------------------------
- Loads saved model & vectorizer
- Classifies user-entered messages
"""

import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Index of "spam" for probability lookup
spam_index = list(model.classes_).index("spam")
threshold = 0.3  # Same threshold used in training

print("Spam SMS Classifier (type 'quit' to exit)")
while True:
    msg = input("\nEnter a message: ")
    if msg.lower() == "quit":
        break
    msg_tfidf = vectorizer.transform([msg])
    prob = model.predict_proba(msg_tfidf)[0][spam_index]
    label = "spam" if prob >= threshold else "ham"
    print(f"Prediction: {label} (spam probability: {prob:.2%})")
