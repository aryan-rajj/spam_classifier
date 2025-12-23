import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------ UI ------------------
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Paste the email content below and check whether it is **Spam** or **Not Spam**.")

# ------------------ Load Dataset ------------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Adjust columns if needed
if "v1" in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ------------------ Train Model ------------------
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ------------------ User Input ------------------
email_text = st.text_area(
    "✉️ Enter Email Content Here",
    height=200,
    placeholder="Paste subject + email body here..."
)

# ------------------ Prediction ------------------
if st.button("Check Spam"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter email content.")
    else:
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)[0]

        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM (HAM)**")
