import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def load_data(filepath):
    print(f"📥 Loading labeled RFM data from: {filepath}")
    return pd.read_csv(filepath)

def train_model(df):
    print("🧠 Training model...")

    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['WillBuy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluation
    print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model to disk
    joblib.dump(model, "purchase_prediction/model_logistic.pkl")
    print("💾 Model saved to: purchase_prediction/model_logistic.pkl")

    return model

def main():
    input_file = "purchase_prediction/data/labeled_rfm.csv"

    if not os.path.exists(input_file):
        print(f"❌ ERROR: Labeled RFM file not found at {input_file}")
        return

    df = load_data(input_file)
    train_model(df)

if __name__ == "__main__":
    main()
