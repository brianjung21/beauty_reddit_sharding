import csv
import re
import time
import unicodedata
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import praw
import prawcore


CLIENT_ID = "GHBnVWNi9oton50-H7IRMQ"
CLIENT_SECRET = "DCgsCWWyDq1UUS629RFl7Wxy8IXnyA"
USER_AGENT = "research"

VALID_SUBREDDITS_CSV = Path("../daily_extraction/files/beauty_subreddits_30.csv")  # column: subreddit (e.g., r/ramen)
KEYWORDS_CSV = Path("../daily_extraction/files/beauty_keywords_with_aliases.csv")

START_DATE_STR = '2025-09-09'
END_DATE_STR = '2025-09-15'

SLEEP_BETWEEN_SUBREDDITS = 1.0
SLEEP_BETWEEN_ERRORS = 2.0

OUT_DIR = Path("new_data")
ALIAS_DAILY_OUT = OUT_DIR / f"beauty_alias_brand_daily_counts_{START_DATE_STR}_{END_DATE_STR}.csv"
RAW_HITS_OUT = OUT_DIR / f"reddit_matches_raw_{START_DATE_STR}_{END_DATE_STR}.csv"


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).lower()


def brand_to_pattern(kw: str) -> re.Pattern:
    tokens = [re.escape(t) for t in kw.split() if t]
    if not tokens:
        return re.compile(r"$^")
    if len(tokens) == 1:
        core = tokens[0].replace(r"\ ", r"[-_\.\s]*")
        pat = rf"(?<!\w){core}(?!\w)"
    else:
        sep = r"[-_\.\s]*"
        pat = rf"(?<!\w){sep.join(tokens)}(?!\w)"
    return re.compile(pat, re.IGNORECASE)


def to_utc_ts(d: date) -> int:
    dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp())


def load_subreddits(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing subreddits CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "subreddit" not in df.columns:
        raise ValueError("subreddits CSV must have a 'subreddit' column")
    subs = []
    for raw in df["subreddit"].astype(str):
        s = raw.strip()
        if not s or s.lower() == "nan":
            continue
        if s.startswith("r/"):
            s = s[2:]
        subs.append(s)
    # keep order, drop dups
    seen = set()
    out = []
    for s in subs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_keywords(csv_path: Path) -> Dict[str, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing keywords CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Keyword" not in df.columns:
        raise ValueError("keywords CSV must have 'Keyword' column")
    if "Aliases" not in df.columns:
        df["Aliases"] = ""
    else:
        df["Aliases"] = df["Aliases"].fillna("")
    mapping: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        main = str(r["Keyword"]).strip().lower()
        if not main or main == "nan":
            continue
        alts_raw = [a.strip() for a in str(r["Aliases"]).split("|") if a is not None]
        alts = [a.lower() for a in alts_raw if a and a.lower() != "nan"]
        # Dedup while preserving order; avoid repeating main
        seen = set([main])
        aliases: List[str] = [main]
        for a in alts:
            if a not in seen:
                seen.add(a)
                aliases.append(a)
        mapping[main] = aliases
    if not mapping:
        raise RuntimeError("No keywords loaded.")
    return mapping


def choose_time_filter(start_d: date, end_d: date) -> str:
    """
    Reddit's search time_filter is coarse; we still do exact UTC filtering via created_utc boundaries.
    """
    days = (end_d - start_d).days
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 30:
        return "month"
    if days <= 365:
        return "year"
    return "all"


def crawl_brand_in_subreddit(
    sr,                       # praw.models.Subreddit
    brand: str,
    aliases: List[str],
    start_d: date,
    end_d: date,
    seen_posts_for_brand: Set[str],
    raw_rows: List[dict],
    alias_daily_counter: Dict[Tuple[str, str, str], int],  # (date_iso, brand, alias) -> mentions
):
    """
    Walk subreddit.search(...) newest→older per alias.
    - We apply strict UTC boundaries: keep posts with START_DATE <= created_utc < END_DATE+1
    - First matched alias 'wins' for that post for this brand.
    - Dedupe per brand by Reddit post id across subreddits & aliases.
    """
    start_ts = to_utc_ts(start_d)
    end_ts_exclusive = to_utc_ts(end_d + timedelta(days=1))
    tf = choose_time_filter(start_d, end_d)

    try:
        for alias in aliases:
            # Query Reddit's own match; we still filter by time and dedupe below
            gen = sr.search(query=alias, sort="new", time_filter=tf, limit=None)
            for subm in gen:
                ct = int(getattr(subm, "created_utc", 0) or 0)

                # too new (>= END+1 day) — keep iterating; listings are newest→older
                if ct >= end_ts_exclusive:
                    continue

                # too old (< START) — break alias loop, we’re going older
                if ct < start_ts:
                    break

                pid = subm.id
                if pid in seen_posts_for_brand:
                    continue

                # mark as seen for this brand
                seen_posts_for_brand.add(pid)

                d_iso = datetime.fromtimestamp(ct, tz=timezone.utc).date().isoformat()
                k = (d_iso, brand, alias)
                alias_daily_counter[k] = alias_daily_counter.get(k, 0) + 1

                raw_rows.append({
                    "date": d_iso,
                    "keyword": brand,
                    "alias": alias,
                    "subreddit": subm.subreddit.display_name,
                    "post_id": pid,
                    "title": (subm.title or "").strip(),
                    "author": getattr(subm.author, "name", "") if getattr(subm, "author", None) else "",
                    "score": int(getattr(subm, "score", 0) or 0),
                    "num_comments": int(getattr(subm, "num_comments", 0) or 0),
                    "permalink": f"https://reddit.com{subm.permalink}" if getattr(subm, "permalink", None) else "",
                    "created_utc": ct,
                })

    except prawcore.exceptions.TooManyRequests:
        print(f"    [rate] TooManyRequests in r/{sr.display_name} — sleeping {SLEEP_BETWEEN_ERRORS}s")
        time.sleep(SLEEP_BETWEEN_ERRORS)
    except Exception as e:
        print(f"    [error] r/{sr.display_name}: {e}")
        time.sleep(SLEEP_BETWEEN_ERRORS)


def main():
    # --- Required fixed window ---
    if not START_DATE_STR or not END_DATE_STR:
        raise ValueError("Please set START_DATE_STR and END_DATE_STR (YYYY-MM-DD).")
    start_d = date.fromisoformat(START_DATE_STR)
    end_d   = date.fromisoformat(END_DATE_STR)
    if end_d < start_d:
        raise ValueError("END_DATE_STR must be >= START_DATE_STR.")
    print(f"Window (UTC): {start_d} → {end_d} (inclusive)")

    # Load inputs
    subreddits = load_subreddits(VALID_SUBREDDITS_CSV)
    kw_map = load_keywords(KEYWORDS_CSV)
    print(f"Loaded {len(subreddits)} subreddits, {len(kw_map)} brands.")

    # Reddit client
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    # Accumulators
    raw_rows: List[dict] = []
    alias_daily_counter: Dict[Tuple[str, str, str], int] = {}

    brands = list(kw_map.keys())
    for bi, brand in enumerate(brands, 1):
        aliases = kw_map[brand]
        print(f"\n[Brand] {brand}  ({bi}/{len(brands)})  aliases={aliases}")

        seen_posts_for_brand: Set[str] = set()  # dedupe per brand

        for si, sub in enumerate(subreddits, 1):
            print(f"  - r/{sub}  ({si}/{len(subreddits)})")
            sr = reddit.subreddit(sub)
            crawl_brand_in_subreddit(
                sr=sr,
                brand=brand,
                aliases=aliases,
                start_d=start_d,
                end_d=end_d,
                seen_posts_for_brand=seen_posts_for_brand,
                raw_rows=raw_rows,
                alias_daily_counter=alias_daily_counter,
            )
            time.sleep(SLEEP_BETWEEN_SUBREDDITS)

        print(f"    → collected unique posts for '{brand}': {len(seen_posts_for_brand)}")

    # Build alias-brand daily panel for the fixed window
    print("\nBuilding alias_brand_daily_counts panel...")
    all_dates = [ (start_d + timedelta(days=i)).isoformat()
                  for i in range((end_d - start_d).days + 1) ]
    rows = []
    for brand, aliases in kw_map.items():
        for alias in aliases:
            for d in all_dates:
                k = (d, brand, alias)
                rows.append({
                    "date": d,
                    "keyword": brand,
                    "alias": alias,
                    "mentions": alias_daily_counter.get(k, 0),
                })
    df_panel = pd.DataFrame(rows)

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_panel.to_csv(ALIAS_DAILY_OUT, index=False, encoding="utf-8")
    print(f"Wrote alias daily counts → {ALIAS_DAILY_OUT.resolve()}  (rows={len(df_panel)})")

    with RAW_HITS_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "keyword", "alias", "subreddit", "post_id", "title",
            "author", "score", "num_comments", "permalink", "created_utc"
        ])
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    print(f"Wrote raw matches → {RAW_HITS_OUT.resolve()}  (rows={len(raw_rows)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
