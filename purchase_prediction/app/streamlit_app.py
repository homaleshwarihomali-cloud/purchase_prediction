import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Purchase Prediction", layout="centered")

st.title("🧠 Customer Purchase Prediction App")
st.markdown("Upload an RFM dataset and predict whether a customer will make a purchase.")

# Load the saved logistic regression model
@st.cache_resource
def load_model():
    return joblib.load("purchase_prediction/model_logistic.pkl")

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload RFM CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Preview of Uploaded Data")
    st.write(df.head())

    required_cols = {'Recency', 'Frequency', 'Monetary'}

    if required_cols.issubset(df.columns):
        # Make predictions
        X = df[['Recency', 'Frequency', 'Monetary']]
        predictions = model.predict(X)
        df['WillBuy_Prediction'] = predictions

        st.subheader("🎯 Prediction Results")
        st.write(df)

        # Download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Results as CSV",
            data=csv,
            file_name='rfm_predictions.csv',
            mime='text/csv'
        )

        # Optional: Show class distribution
        st.subheader("📊 Summary")
        counts = df['WillBuy_Prediction'].value_counts().rename({0: "❌ Won't Buy", 1: "✅ Will Buy"})
        st.bar_chart(counts)

    else:
        st.error("❗ The uploaded file must contain 'Recency', 'Frequency', and 'Monetary' columns.")
