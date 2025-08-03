import pandas as pd
import os

def load_data(filepath):
    print(f"📥 Loading data from: {filepath}")
    df = pd.read_excel(filepath)
    print("✅ Loaded! Initial shape:", df.shape)
    return df

def clean_data(df):
    print("🧹 Cleaning data...")
    df = df.dropna(subset=['CustomerID'])       # Drop rows without CustomerID
    df = df[df['Quantity'] > 0]                 # Remove negative Quantity (returns)
    df = df[df['UnitPrice'] > 0]                # Remove zero or invalid prices

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print("✅ Cleaned! Final shape:", df.shape)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned data saved to: {output_path}")

def main():
    input_file = "purchase_prediction/data/Online Retail.xlsx"  # use Excel file
    output_file = "purchase_prediction/data/cleaned_data.csv"

    if not os.path.exists(input_file):
        print(f"❌ ERROR: File not found at {input_file}")
        return

    df = load_data(input_file)
    cleaned_df = clean_data(df)
    save_cleaned_data(cleaned_df, output_file)

if __name__ == "__main__":
    main()
