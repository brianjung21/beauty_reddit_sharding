from pathlib import Path
import hashlib
from typing import List
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

DATA_DIR = Path("combined_data")
RAW_FILE = "reddit_matches_raw_combined.csv"
CACHE_FILE = "sentiment_cache.csv"

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

BATCH_SIZE = 64
MAX_LENGTH = 256
CONCAT_SELFTEXT = True
ADD_VADER = False

_NLTK_AVAILABLE = False
if ADD_VADER:
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        _NLTK_AVAILABLE = True
    except Exception:
        print("[warn] NLTK/VADER not available.")


def stable_post_id(row: pd.Series) -> str:
    """Stable dedup key built from (title, date, subreddit, keyword)."""
    title = str(row.get("title", "")).strip()
    date = str(row.get("date", ""))
    sub = str(row.get("subreddit", "")).strip()
    kw = str(row.get("keyword", "")).strip()
    s = f"{title}\t{date}\t{sub}\t{kw}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_raw(path: Path, concat_selftext: bool) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    keep_candidates = ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments", "selftext", "body"]
    cols = [c for c in keep_candidates if c in df.columns]
    df = df[cols].copy()

    # Normalize text fields
    df["title"] = df.get("title", "").astype(str).fillna("").str.strip()

    if concat_selftext:
        text2 = None
        if "selftext" in df.columns:
            text2 = df["selftext"].astype(str)
        elif "body" in df.columns:
            text2 = df["body"].astype(str)
        if text2 is not None:
            df["title"] = (df["title"] + " " + text2.fillna("")).str.strip()

    # Drop empty titles
    df = df[df["title"] != ""].copy()

    # Construct post_id and drop dupes
    df["post_id"] = df.apply(stable_post_id, axis=1)
    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    return df


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        print("[warn] Failed to read cache; starting fresh.")
        return pd.DataFrame()


def build_pipeline(model_name: str, device_arg: int, max_length: int, force_mps: bool = False) -> TextClassificationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # If running on Apple Silicon MPS, move the PyTorch model to the MPS device, and keep pipeline device=-1
    if force_mps:
        model.to(torch.device("mps"))
        device_arg = -1  # pipeline uses CPU code-path, but model is on MPS
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device_arg,
        return_all_scores=True,
        truncation=True,
        max_length=max_length,
    )
    return pipe


def score_transformer(texts: List[str], pipe: TextClassificationPipeline, batch_size: int) -> pd.DataFrame:
    out_rows = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        chunk = texts[i:i + batch_size]
        outs = pipe(chunk)
        for rec in outs:
            # rec: list of dicts with 'label' and 'score'
            probs = {d["label"].lower(): float(d["score"]) for d in rec}
            neg = probs.get("negative", probs.get("neg", 0.0))
            neu = probs.get("neutral", probs.get("neu", 0.0))
            pos = probs.get("positive", probs.get("pos", 0.0))
            compound = pos - neg  # simple symmetric proxy in [-1, 1]
            out_rows.append({"neg": neg, "neu": neu, "pos": pos, "compound": compound})
    return pd.DataFrame(out_rows)


def ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def maybe_add_vader(cache_df: pd.DataFrame) -> pd.DataFrame:
    if not ADD_VADER:
        return cache_df
    if not _NLTK_AVAILABLE:
        print("[vader] Skipping VADER (NLTK not available).")
        return cache_df
    # ensure lexicon
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        import nltk as _n
        _n.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    if "compound_vader" not in cache_df.columns:
        cache_df["compound_vader"] = np.nan
    mask = cache_df["compound_vader"].isna() & cache_df["title"].notna()
    if mask.any():
        titles = cache_df.loc[mask, "title"].astype(str).tolist()
        vals = [sia.polarity_scores(t)["compound"] for t in tqdm(titles, desc="VADER")]
        cache_df.loc[mask, "compound_vader"] = vals
    return cache_df


def main():
    data_dir = DATA_DIR
    raw_path = data_dir / RAW_FILE
    cache_path = data_dir / CACHE_FILE

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    print(f"[info] Loading raw:   {raw_path}")
    raw = load_raw(raw_path, concat_selftext=CONCAT_SELFTEXT)

    print(f"[info] Loading cache: {cache_path}")
    cache = load_cache(cache_path)

    have_ids = set([] if cache.empty else cache["post_id"].astype(str).tolist())
    todo = raw[~raw["post_id"].astype(str).isin(have_ids)].copy()

    print(f"[stats] total raw unique posts: {len(raw):,}")
    print(f"[stats] already cached:         {len(raw) - len(todo):,}")
    print(f"[stats] to score now:           {len(todo):,}")

    use_cuda = torch.cuda.is_available()
    use_mps = (not use_cuda) and torch.backends.mps.is_available()
    if use_cuda:
        device_name = "CUDA:0"
        device_arg = 0
        force_mps = False
    elif use_mps:
        device_name = "MPS"
        device_arg = -1  # pipeline keeps CPU code-path; model is moved to MPS inside build_pipeline
        force_mps = True
    else:
        device_name = "CPU"
        device_arg = -1
        force_mps = False
    print(f"[info] Using model: {MODEL_NAME} | device: {device_name}")

    if len(todo) > 0:
        pipe = build_pipeline(MODEL_NAME, device_arg=device_arg, max_length=MAX_LENGTH, force_mps=force_mps)
        texts = todo["title"].astype(str).tolist()
        scored = score_transformer(texts, pipe, batch_size=BATCH_SIZE)
        out = pd.concat([todo.reset_index(drop=True), scored], axis=1)
        out["model_name"] = MODEL_NAME
        out["text_len"] = out["title"].astype(str).str.len()

        # Preserve score/num_comments if present in raw
        desired = [
            "post_id", "date", "keyword", "subreddit", "title", "text_len",
            "neg", "neu", "pos", "compound", "model_name"
        ]
        for opt in ["score", "num_comments"]:
            if opt in out.columns:
                desired.append(opt)
        out = out[desired].copy()
    else:
        out = pd.DataFrame(columns=[
            "post_id", "date", "keyword", "subreddit", "title", "text_len",
            "neg", "neu", "pos", "compound", "model_name", "score", "num_comments"
        ])

    # Merge with existing cache & dedup
    if cache.empty:
        merged = out.copy()
    else:
        all_cols = list(set(out.columns) | set(cache.columns))
        cache = ensure_columns(cache, all_cols)
        out = ensure_columns(out, all_cols)
        merged = pd.concat([cache, out], ignore_index=True)
        merged = merged.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    # Optional VADER
    merged = maybe_add_vader(merged)

    merged.sort_values("date", inplace=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    print(f"[done] Saved: {cache_path}  ({len(merged):,} rows)")


if __name__ == "__main__":
    main()