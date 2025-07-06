import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_ham_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("📩 Spam vs Ham Classifier")
st.subheader("Detect whether a message is spam or not using ML!")

# User input
user_input = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # Vectorize the input text
        vectorized_input = vectorizer.transform([user_input])
        
        #prediction
        prediction = model.predict(vectorized_input)[0]

        # Display result
        if prediction == 1:
            st.error("🚫 Spam Message Detected!")
        else:
            st.success("✅ This is a Ham (Not Spam) Message.")
