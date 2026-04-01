import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -----------------------------
# LOAD DATA
# -----------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")


# Add labels
fake["label"] = 0
true["label"] = 1
print(fake.columns)

# Normalize columns
fake.columns = [c.lower() for c in fake.columns]
true.columns = [c.lower() for c in true.columns]

# Use only text + label
fake = fake[["text", "label"]]
true = true[["text", "label"]]


# Combine
df = pd.concat([fake, true]).sample(frac=1, random_state=42)

# Clean
df["text"] = df["text"].apply(clean_text)

# -----------------------------
# BALANCE DATASET
# -----------------------------
fake_df = df[df.label == 0]
real_df = df[df.label == 1]

min_size = min(len(fake_df), len(real_df))

fake_balanced = resample(fake_df, replace=False, n_samples=min_size, random_state=42)
real_balanced = resample(real_df, replace=False, n_samples=min_size, random_state=42)

df_balanced = pd.concat([fake_balanced, real_balanced]).sample(frac=1, random_state=42)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["text"], df_balanced["label"], test_size=0.2, random_state=42
)

# -----------------------------
# TF-IDF + LOGISTIC REGRESSION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
preds = model.predict(X_test_vec)
y_pred = preds

print("Accuracy:", accuracy_score(y_test, preds))

# -----------------------------
# SAVE MODEL
# -----------------------------
with open("ml/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("ml/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


print("New model trained and saved successfully.")
