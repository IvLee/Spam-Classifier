# 📩 Spam SMS Classifier

A machine learning project that detects spam SMS messages using the UCI SMS Spam Collection dataset.

## 🚀 Features
- Preprocessing (lowercase, punctuation removal, stopword removal, stemming)
- TF-IDF text vectorization
- Model comparison (Naive Bayes & Logistic Regression)
- Saves best model to disk
- CLI prediction tool
- Streamlit web app for single & batch classification
- Supports CSV, TXT, and Excel (.xlsx) batch uploads
- Downloadable prediction results with original filename

## 📂 Project Structure
```
spam-classifier/
│
├── src/
│   ├── train_model.py       # Train & save best model
│   ├── predict.py           # Command-line predictions
│   ├── app.py               # Streamlit web app
├── requirements.txt
├── README.md
├── spam_classifier.pkl      # Saved best model
├── vectorizer.pkl           # Saved TF-IDF vectorizer
```

## ⚙️ Installation
```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model
Run this to train the model and save it to disk:
```bash
python src/train_model.py
```

## 🖥 CLI Prediction
Run the interactive CLI classifier:
```bash
python src/predict.py
```
Type messages and get spam/ham predictions instantly.

## 🌐 Streamlit Web App (Local)
Run the web app locally:
```bash
streamlit run src/app.py
```

## ☁️ Streamlit Cloud Deployment
You can try the app online here:
[**Live Demo on Streamlit Cloud**](https://spam-or-ham-spam-classifier.streamlit.app/)

## 📊 Dataset
[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

## 📜 License
MIT License
