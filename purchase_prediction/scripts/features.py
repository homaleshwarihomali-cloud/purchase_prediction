import pandas as pd
import os

def load_cleaned_data(filepath):
    print(f"📥 Loading cleaned data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["InvoiceDate"])
    return df

def generate_rfm(df):
    print("🧠 Generating RFM features...")

    # Reference date: day after the last transaction
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Group by CustomerID to calculate Recency, Frequency, Monetary
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,   # Recency
        'InvoiceNo': 'nunique',                                     # Frequency
        'TotalPrice': 'sum'                                         # Monetary
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    print("✅ RFM features generated!")
    return rfm

def save_rfm(rfm, output_path):
    rfm.to_csv(output_path, index=False)
    print(f"💾 RFM features saved to: {output_path}")

def main():
    input_file = "purchase_prediction/data/cleaned_data.csv"
    output_file = "purchase_prediction/data/rfm_features.csv"

    if not os.path.exists(input_file):
        print(f"❌ ERROR: Cleaned data not found at {input_file}")
        return

    df = load_cleaned_data(input_file)
    rfm = generate_rfm(df)
    save_rfm(rfm, output_file)

if __name__ == "__main__":
    main()
