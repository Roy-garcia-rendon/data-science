# src/etl_basic.py
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "sales.csv"
PROC = Path(__file__).resolve().parents[1] / "data" / "processed" / "sales_clean.csv"

def load_raw():
    df = pd.read_csv(RAW)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates()
    # Filtros plausibles
    df = df[df["price"].between(0.5, 10000)]
    df = df[df["quantity"].between(1, 100)]
    df["discount"] = df["discount"].clip(0, 0.9)
    df["revenue"] = df["price"] * df["quantity"] * (1 - df["discount"])
    return df

def run():
    df = load_raw()
    df = clean(df)
    PROC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC, index=False)
    print(f"Guardado: {PROC} ({len(df)} filas)")

if __name__ == "__main__":
    run()
