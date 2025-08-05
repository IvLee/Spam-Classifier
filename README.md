# 📩 Spam SMS Classifier

A machine learning project that detects spam SMS messages using the UCI SMS Spam Collection dataset.

## 🚀 Features
- Preprocessing (lowercase, punctuation removal, stopword removal, stemming)
- TF-IDF text vectorization
- Model comparison (Naive Bayes & Logistic Regression)
- Saves best model to disk
- CLI prediction tool
- Streamlit web app for easy use

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
```

## ⚙️ Installation
```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model
```bash
python src/train_model.py
```

## 🖥 CLI Prediction
```bash
python src/predict.py
```
Type messages and get spam/ham predictions.

## 🌐 Run the Web App
```bash
streamlit run src/app.py
```

## 📊 Dataset
[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

## 📜 License
MIT License
