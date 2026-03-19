import pandas as pd
import os
from config import DATA_DIR

def load_csv(csv_path):
    """Load and analyze a CSV dataset."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded: {os.path.basename(csv_path)}")
        print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def summarize_dataset(df):
    """Generate a text summary of the dataset for AI analysis."""
    summary = []

    summary.append(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    summary.append(f"Columns: {', '.join(df.columns.tolist())}")
    summary.append(f"\nData Types:\n{df.dtypes.to_string()}")
    summary.append(f"\nBasic Statistics:\n{df.describe().to_string()}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        summary.append(f"\nMissing Values:\n{missing[missing > 0].to_string()}")
    else:
        summary.append("\nNo missing values found.")

    # Sample rows
    summary.append(f"\nSample Data (first 5 rows):\n{df.head().to_string()}")

    full_summary = "\n".join(summary)
    print("✅ Dataset summary generated!")
    return full_summary

def csv_to_text(csv_path):
    """Convert CSV file to text summary for embedding."""
    df = load_csv(csv_path)
    if df is None:
        return None

    summary = summarize_dataset(df)
    return {
        "filename": os.path.basename(csv_path),
        "content": summary,
        "dataframe": df
    }

if __name__ == "__main__":
    # Create a sample CSV for testing
    sample_data = {
        "experiment": ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"],
        "attention_heads": [4, 8, 16, 32, 64],
        "bleu_score": [25.3, 27.8, 28.4, 28.1, 27.2],
        "training_time_hours": [12, 18, 24, 36, 48],
        "parameters_million": [45, 65, 95, 145, 210]
    }

    df = pd.DataFrame(sample_data)
    sample_path = os.path.join(DATA_DIR, "transformer_experiments.csv")
    df.to_csv(sample_path, index=False)
    print(f"✅ Sample CSV created at: {sample_path}")

    # Test loading
    result = csv_to_text(sample_path)
    if result:
        print("\n📊 Dataset Summary:")
        print(result["content"])