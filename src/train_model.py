"""
Train and save the best Spam SMS Classifier (with preprocessing & model comparison)
-----------------------------------------------------------------------------------
- Loads dataset
- Cleans and normalizes text
- Trains Naive Bayes and Logistic Regression
- Compares accuracy
- Saves the best model & vectorizer
"""

import pandas as pd
import re
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (run once)
nltk.download('stopwords')

# ----------------------------
# 1. Preprocessing Function
# ----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ----------------------------
# 2. Load Dataset
# ----------------------------
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=["label", "message"])

print(f"Dataset loaded: {df.shape[0]} messages")
print(df['label'].value_counts(), "\n")

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# ----------------------------
# 3. Train/Test Split
# ----------------------------
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Vectorization
# ----------------------------
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------
# 5. Train & Evaluate Multiple Models
# ----------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

best_model = None
best_accuracy = 0

for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = clf

# ----------------------------
# 6. Save Best Model & Vectorizer
# ----------------------------
joblib.dump(best_model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print(f"\nBest model ({best_model.__class__.__name__}) saved with accuracy {best_accuracy:.2%}!")
