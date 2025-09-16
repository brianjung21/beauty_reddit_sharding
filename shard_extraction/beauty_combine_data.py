from pathlib import Path
import pandas as pd

OLD_DIR = Path("data")
NEW_DIR = Path("new_data")
OUT_DIR = Path("combined_data")

DAILY_PATTERNS = [
    "beauty_alias_brand_daily_counts*.csv",
]
RAW_PATTERNS = [
    "reddit_matches_raw*.csv",
]


def find_files(folder: Path, patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))
    return files

def read_df(path: Path, parse_dates=None):
    parse_dates = parse_dates or []
    # Only parse date-like columns that exist
    cols = list(pd.read_csv(path, nrows=0).columns)
    parse = [c for c in parse_dates if c in cols]
    return pd.read_csv(path, parse_dates=parse)

def load_many(folder: Path, patterns, label: str, parse_dates=None):
    fps = find_files(folder, patterns)
    dfs = []
    for fp in fps:
        try:
            df = read_df(fp, parse_dates=parse_dates)
            dfs.append(df)
            print(f"[ok] Loaded {label}: {fp}  rows={len(df):,}")
        except Exception as e:
            print(f"[warn] Failed to load {fp}: {e}")
    if not dfs:
        print(f"[info] No {label} files found in {folder}")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def brief(df, name, date_cols=None, key_cols=None):
    print(f"\n[{name}] rows={len(df):,}")
    if df.empty:
        return
    # date ranges
    for c in (date_cols or []):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            print(f"  {c}: {s.min()} → {s.max()}")
    # dup check
    if key_cols and all(k in df.columns for k in key_cols):
        dups = df.duplicated(subset=key_cols).sum()
        print(f"  duplicates on {key_cols}: {dups}")
    print("  columns:", list(df.columns))

def combine_daily(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    need_cols = {"date", "keyword", "alias", "mentions"}
    if old_df.empty and new_df.empty:
        return pd.DataFrame(columns=list(need_cols))
    df = pd.concat([old_df, new_df], ignore_index=True)
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"Daily counts missing columns: {missing}. Have: {list(df.columns)}")
    # Coerce mentions to int (safe fill)
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce").fillna(0).astype(int)
    # Keep MAX per (date, keyword, alias)
    out = (df.groupby(["date", "keyword", "alias"], as_index=False)["mentions"]
             .max()
             .sort_values(["date", "keyword", "alias"]))
    return out

def combine_raw(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df.empty and new_df.empty:
        return pd.DataFrame(columns=["date","keyword","alias","subreddit","post_id","title","author","score","num_comments","permalink","created_utc"])
    df = pd.concat([old_df, new_df], ignore_index=True)
    # Minimal column presence check (post_id + keyword for dedup)
    for c in ["post_id", "keyword"]:
        if c not in df.columns:
            raise KeyError(f"Raw matches missing required column: {c}")
    # Dedup by (post_id, keyword); keep first
    df = df.drop_duplicates(subset=["post_id", "keyword"], keep="first")
    # Optional: normalize dtypes
    for c in ["score", "num_comments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def main():
    print("=== Loading old and new ===")
    old_daily = load_many(OLD_DIR, DAILY_PATTERNS, "daily(old)", parse_dates=["date"])
    new_daily = load_many(NEW_DIR, DAILY_PATTERNS, "daily(new)", parse_dates=["date"])
    old_raw   = load_many(OLD_DIR, RAW_PATTERNS,   "raw(old)",   parse_dates=["date"])
    new_raw   = load_many(NEW_DIR, RAW_PATTERNS,   "raw(new)",   parse_dates=["date"])

    brief(old_daily, "old_daily", date_cols=["date"], key_cols=["date","keyword","alias"])
    brief(new_daily, "new_daily", date_cols=["date"], key_cols=["date","keyword","alias"])
    brief(old_raw,   "old_raw",   date_cols=["date"], key_cols=["post_id","keyword"])
    brief(new_raw,   "new_raw",   date_cols=["date"], key_cols=["post_id","keyword"])

    print("\n=== Combining ===")
    daily_combined = combine_daily(old_daily, new_daily)
    raw_combined   = combine_raw(old_raw, new_raw)

    brief(daily_combined, "daily_combined", date_cols=["date"], key_cols=["date","keyword","alias"])
    brief(raw_combined,   "raw_combined",   date_cols=["date"], key_cols=["post_id","keyword"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUT_DIR / "beauty_alias_brand_daily_counts_combined.csv"
    raw_path   = OUT_DIR / "reddit_matches_raw_combined.csv"

    daily_combined.to_csv(daily_path, index=False, encoding="utf-8")
    raw_combined.to_csv(raw_path, index=False, encoding="utf-8")

    print(f"\n[ok] Wrote daily → {daily_path.resolve()} (rows={len(daily_combined):,})")
    print(f"[ok] Wrote raw   → {raw_path.resolve()} (rows={len(raw_combined):,})")

    # Quick final ranges
    if not daily_combined.empty:
        s = pd.to_datetime(daily_combined["date"], errors="coerce", utc=True)
        print(f"daily date range: {s.min()} → {s.max()}")
    if "date" in raw_combined.columns and not raw_combined.empty:
        s2 = pd.to_datetime(raw_combined["date"], errors="coerce", utc=True)
        print(f"raw date range:   {s2.min()} → {s2.max()}")

if __name__ == "__main__":
    main()