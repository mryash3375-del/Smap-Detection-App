import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Config
st.set_page_config(page_title="Spam Detection App", layout="wide")

# Title
st.title("📩 SMS Spam Detection System")
st.markdown("Use this app to check if an SMS message is **Spam** or **Ham** (Legitimate).")

# Sidebar
st.sidebar.header("User Input")

# 1. Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
        df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
        df.drop_duplicates(inplace=True)
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Error: 'spam.csv' not found. Please make sure the dataset is in the same directory.")
    st.stop()

# 2. Train Model (Cached)
@st.cache_resource
def train_model(df):
    X = df['message']
    y = df['label_num']
    
    # Vectorization
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Model (Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, tfidf, acc, y_test, y_pred

model, tfidf, accuracy, y_test, y_pred = train_model(df)

# Sidebar Info
st.sidebar.success(f"Model Accuracy: **{accuracy:.2%}**")

# 3. Main Interface - Prediction
user_input = st.text_area("Enter an SMS message to classify:", height=150)

if st.button("Predict"):
    if user_input:
        data = tfidf.transform([user_input])
        prediction = model.predict(data)[0]
        
        if prediction == 1:
            st.error("🚨 This message is likely **SPAM**!")
        else:
            st.success("✅ This message is likely **HAM** (Legitimate).")
    else:
        st.warning("Please enter a message to classify.")

st.markdown("---")

# 4. EDA Section
if st.checkbox("Show Dataset & Statistics"):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spam vs Ham Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='label', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax_cm)
        st.pyplot(fig_cm)
