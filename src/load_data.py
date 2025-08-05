# src/load_data.py
import pandas as pd

# URL to dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Load data into a DataFrame
df = pd.read_table(url, header=None, names=["label", "message"])

# Show first few rows
print(df.head())

# Show shape and class distribution
print("\nShape of dataset:", df.shape)
print("\nLabel distribution:")
print(df['label'].value_counts())
