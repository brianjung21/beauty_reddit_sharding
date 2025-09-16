"""
This script is a trial for adding alias variations per brand
and then deduplicating by post ID before counting.
Also keeps the top_subreddits column.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import praw
import time
import re
import unicodedata
import prawcore

CLIENT_ID = "loser"
CLIENT_SECRET = "loser"
USER_AGENT = "loser"

KEYWORDS_PATH = Path("files/beauty_keywords_with_aliases.csv")
OUTPUT_PATH = Path("data/beauty_alias_brand_daily_counts.csv")

SEARCH_LIMIT = None
SLEEP_BETWEEN_QUERIES = 1

REQUEST_TIMEOUT = 30  # seconds for HTTP timeout to Reddit API
MAX_SEARCH_RETRIES = 5  # exponential backoff retries per keyword


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def brand_to_pattern(kw: str) -> re.Pattern:
    tokens = [re.escape(t) for t in kw.split() if t]
    if not tokens:
        return re.compile(r"$^")
    if len(tokens) == 1:
        core = tokens[0].replace(r"\ ", r"[-_\.\s]*")
        pattern = rf"(?<!\w){core}(?!\w)"
    else:
        sep = r"[-_\.\s]*"
        core = sep.join(tokens)
        pattern = rf"(?<!\w){core}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def load_keywords(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Keyword" not in df.columns:
        raise ValueError("Expected a header column named 'Keyword'")
    if "Aliases" not in df.columns:
        df["Aliases"] = ""
    mapping = {}
    for _, r in df.iterrows():
        main = str(r["Keyword"]).strip().lower()
        if not main:
            continue
        alts = [a.strip().lower() for a in str(r["Aliases"]).split("|") if a.strip()]
        mapping[main] = [main] + alts
    return mapping


def utc_date_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


# Retry logic for transient Reddit API errors
def iter_search_with_retries(subreddit, query: str, limit=None, max_retries: int = MAX_SEARCH_RETRIES):
    """Yield submissions from subreddit.search with retry/backoff on transient errors.
    Safe to re-run because downstream de-duplicates by post_id.
    """
    attempts = 0
    while True:
        try:
            for subm in subreddit.search(query, sort="new", limit=limit):
                yield subm
            break  # completed without error
        except (prawcore.exceptions.RequestException,
                prawcore.exceptions.ServerError,
                prawcore.exceptions.ResponseException) as e:
            attempts += 1
            if attempts > max_retries:
                print(f"[WARN] search failed for query='{query}' after {attempts-1} retries: {e}")
                raise
            sleep_s = min(60, 2 ** attempts)
            print(f"[INFO] transient error on query='{query}': {e} → retrying in {sleep_s}s (attempt {attempts}/{max_retries})")
            time.sleep(sleep_s)


def main():
    kw_map = load_keywords(KEYWORDS_PATH)
    if not kw_map:
        print("No keywords found in CSV")
        return

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        requestor_kwargs={"timeout": REQUEST_TIMEOUT},
    )

    patterns = {main: [brand_to_pattern(v) for v in variants] for main, variants in kw_map.items()}

    rows = []

    for main_kw in kw_map.keys():
        print(f"Searching for keyword: {main_kw}")
        subreddit = reddit.subreddit('all')
        matched = 0
        try:
            for submission in iter_search_with_retries(subreddit, main_kw, limit=SEARCH_LIMIT):
                text_raw = f"{submission.title or ''}\n{submission.selftext or ''}"
                text_norm = normalize_text(text_raw)
                if any(p.search(text_norm) for p in patterns[main_kw]):
                    the_date = utc_date_from_timestamp(submission.created_utc)
                    rows.append({
                        "date": the_date,
                        "keyword": main_kw,
                        "subreddit": submission.subreddit.display_name,
                        "post_id": submission.id
                    })
                    matched += 1
        except Exception as e:
            # Don’t abort the entire run for one problematic keyword
            print(f"[ERROR] Skipping keyword '{main_kw}' due to repeated errors: {e}")
        print(f"    -> {matched} posts matched")
        time.sleep(SLEEP_BETWEEN_QUERIES)

    if rows:
        df_tmp = pd.DataFrame(rows)
        df_tmp = df_tmp.drop_duplicates(subset=["date", "keyword", "post_id"])
        post_counts = (
            df_tmp.groupby(["date", "keyword"], as_index=False).size()
            .rename(columns={"size": "post_mentions"})
        )

        top_subs = (
            df_tmp.groupby(["date", "keyword"])['subreddit']
            .apply(lambda s: ";".join(s.value_counts().head(3).index.tolist()))
            .reset_index(name="top_subreddits")
        )
        df_out = post_counts.merge(top_subs, on=["date", "keyword"], how="left")
    else:
        df_out = pd.DataFrame(columns=["date", "keyword", "post_mentions", "top_subreddits"])

    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Done. Wrote: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
