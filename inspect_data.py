from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

files = list(DATA_DIR.rglob("*.csv"))

if not files:
    raise FileNotFoundError(
        "No CSV files found. Put the dataset CSVs inside data/."
    )

print(f"Found {len(files)} CSV files")

for file in files:
    print(f"\nFile: {file.name}")

    df = pd.read_csv(file, nrows=5, low_memory=False)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    print("Shape of sample:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample rows:")
    print(df.head(2).to_string(index=False))

    label_columns = [
        col for col in df.columns
        if col.lower() == "label"
    ]

    if label_columns:
        print("Sample labels:")
        print(df[label_columns[0]].value_counts())