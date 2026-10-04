import io
import sys
import pandas as pd

# ---------------------------------------------------------------
# 1. LOAD  (local file path OR a web URL)
# ---------------------------------------------------------------
DEFAULT_SOURCE = "bedtime_screentime_sleep_debt.csv"   # same folder as this script


def clean_url(url: str) -> str:
    """Turn a normal GitHub page link into a raw-file link."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def load_data(source: str) -> pd.DataFrame:
    source = source.strip().strip("'\"")           # handles drag-and-drop paths in Terminal
    if source.startswith("http"):
        import requests
        resp = requests.get(clean_url(source), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    return pd.read_csv(source)


source = sys.argv[1] if len(sys.argv) > 1 else input(
    f"CSV path or URL (Enter for '{DEFAULT_SOURCE}'): ").strip() or DEFAULT_SOURCE
df = load_data(source)

# ---------------------------------------------------------------
# 2. INSPECT
# ---------------------------------------------------------------
print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nDataset Information:")
df.info()
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# 3. CLEAN
# ---------------------------------------------------------------
# Missing values (none in this file, but kept so the script is safe to reuse)
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(exclude="number").columns.drop("user_id")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Duplicates
df = df.drop_duplicates()

# Tidy text values
for col in cat_cols:
    df[col] = df[col].str.strip()

# ---------------------------------------------------------------
# 4. DATA TYPES
# ---------------------------------------------------------------
for col in ["gender", "occupation_type", "chronotype", "primary_bedtime_app"]:
    df[col] = df[col].astype("category")

df["blue_light_filter_active"] = df["blue_light_filter_active"].astype(bool)

# Ordered category: worst -> best
debt_order = ["Severe Sleep Debt", "Moderate Debt", "Mild Deficit", "Optimal Recovery"]
df["sleep_debt_category"] = pd.Categorical(df["sleep_debt_category"], categories=debt_order, ordered=True)

# ---------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["age_group"] = pd.cut(
    df["age"], bins=[17, 24, 34, 44, 54, 65],
    labels=["18-24", "25-34", "35-44", "45-54", "55-65"],
)

df["phone_use_group"] = pd.cut(
    df["bedtime_phone_minutes"], bins=[0, 30, 60, 120, 180],
    labels=["Low (<=30m)", "Moderate (31-60m)", "High (61-120m)", "Very High (>120m)"],
)

df["sleep_deficit_hours"] = (8 - df["total_sleep_hours"]).clip(lower=0).round(2)   # vs 8h target
df["deep_sleep_hours"] = (df["total_sleep_hours"] * df["deep_sleep_pct"] / 100).round(2)
df["rem_sleep_hours"] = (df["total_sleep_hours"] * df["rem_sleep_pct"] / 100).round(2)
df["late_caffeine"] = df["caffeine_post_5pm_mg"] > 0
df["slow_to_fall_asleep"] = df["sleep_latency_min"] > 30   # common clinical cutoff

# ---------------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------------
print("\nProcessed Data:")
print(df.head())
print("\nNew shape:", df.shape)

df.to_csv("sleep_debt_preprocessed.csv", index=False)
print("\nPreprocessing completed. Saved as sleep_debt_preprocessed.csv")
