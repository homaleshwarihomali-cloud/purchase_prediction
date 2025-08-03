import pandas as pd
import os

def load_rfm(filepath):
    print(f"📥 Loading RFM data from: {filepath}")
    return pd.read_csv(filepath)

def label_customers(rfm):
    print("🏷️  Labelling customers...")

    # Simple rule-based labelling logic:
    rfm['WillBuy'] = 0  # default = not likely to buy

    # Rule: customer is likely to buy if:
    # Recency is LOW (recent purchase)
    # Frequency is HIGH (active customer)
    # Monetary is HIGH (spends a lot)
    rfm.loc[
        (rfm['Recency'] <= 30) &
        (rfm['Frequency'] >= 5) &
        (rfm['Monetary'] >= 100),
        'WillBuy'
    ] = 1

    print("✅ Labels added. Class counts:")
    print(rfm['WillBuy'].value_counts())

    return rfm

def save_labeled_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"💾 Labeled data saved to: {output_path}")

def main():
    input_file = "purchase_prediction/data/rfm_features.csv"
    output_file = "purchase_prediction/data/labeled_rfm.csv"

    if not os.path.exists(input_file):
        print(f"❌ ERROR: File not found at {input_file}")
        return

    rfm = load_rfm(input_file)
    labeled = label_customers(rfm)
    save_labeled_data(labeled, output_file)

if __name__ == "__main__":
    main()
