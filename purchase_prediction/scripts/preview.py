import pandas as pd

def main():
    file_path = "purchase_prediction/data/cleaned_data.csv"
    df = pd.read_csv(file_path)

    print("📊 Preview of cleaned data:")
    print(df.head(10))  # show first 10 rows

    print("\n🧠 Columns available:")
    print(df.columns.tolist())

    print("\n📈 Summary:")
    print(df.describe())

if __name__ == "__main__":
    main()
