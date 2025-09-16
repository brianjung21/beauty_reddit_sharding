"""
Generates common variants per brand
Spacing/sep variants: "olive young", oliveyoung, olive-young, olive_young, etc.
&<->and swaps: p&g, "p and g", "procter & gamble"
punctuation-stripped
A short acronym if it seems reasonable
Removes obvious noise, deduplicates, and writes a new CSV with
Aliases column.
"""
import re
import unicodedata
from pathlib import Path
import pandas as pd

IN_PATH = Path('files/beauty_keywords.csv')
MANUAL_PATH = Path('files/manual_aliases.csv')  # optional
OUT_PATH = Path("files/beauty_keywords_with_aliases.csv")

STOPWORDS = {"co", "inc", "ltd", "llc", "corp", "company", "official", "store"}
SEPS = ["", "-", "_", "."]


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip().lower()


def tokenize(brand: str):
    s = re.sub(r"[^\w&]+", " ", brand.lower())
    raw = re.split(r"[\s]+", s)
    toks = [t for t in raw if t and t not in STOPWORDS]
    return toks


def and_amp_variants(tokens):
    out = set()
    s = " ".join(tokens)
    out.add(tuple(tokens))
    out.add(tuple(t if t != "&" else "and" for t in tokens))
    out.add(tuple(t if t != "and" else "&" for t in tokens))
    return {tuple(t for t in tup if t) for tup in out}


def join_variants(tokens):
    variants = set()
    for sep in SEPS:
        variants.add(sep.join(tokens))
    return variants


def punctuation_stripped(s: str):
    return re.sub(r"[^\w]+", "", s)


# def maybe_acronym(tokens):
#     core = [t for t in tokens if t not in {"&", "and"}]
#     if not core:
#         return None
#     acro = "".join(t[0] for t in core if t)
#     if 2 <= len(acro) <= 6 and not acro.isdigit():
#         return acro
#     return None


def gen_aliases_for(brand: str):
    brand = norm(brand)
    if not brand:
        return []

    tokens = tokenize(brand)
    if not tokens:
        return []

    variants = set()
    token_sets = and_amp_variants(tokens)
    for tokset in token_sets:
        variants |= join_variants(tokset)

    variants.add(punctuation_stripped(brand))
    # acro = maybe_acronym(tokens)
    # if acro:
    #     variants.add(acro)
    variants = {v for v in variants if v and v != brand}

    return sorted(variants)


def main():
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    if "Keyword" not in df.columns:
        raise ValueError("Expected 'Keyword' column in keywords.csv")
    df['Keyword'] = df['Keyword'].astype(str).map(norm)
    df = df[df['Keyword'].astype(bool)].drop_duplicates(subset=['Keyword']).reset_index(drop=True)
    df['Aliases'] = df['Keyword'].apply(lambda kw: "|".join(gen_aliases_for(kw)))

    if MANUAL_PATH.exists():
        m = pd.read_csv(MANUAL_PATH, encoding="utf-8-sig")
        if {"Keyword", "Aliases"}.issubset(m.columns):
            m["Keyword"] = m["Keyword"].astype(str).map(norm)
            m["Aliases"] = m["Aliases"].fillna("").astype(str)
            df = df.merge(m[["Keyword", "Aliases"]], on="Keyword", how='left', suffixes=("", "_manual"))

            def combine(a, b):
                parts = [p for p in (str(a).split("|") if pd.notna(a) else []) if p]
                parts += [p for p in (str(b).split("|") if pd.notna(b) else []) if p]
                parts = sorted({p for p in parts if p and p != ""})
                return "|".join(parts)
            df["Aliases"] = df.apply(lambda r: combine(r["Aliases"], r["Aliases_manual"]), axis=1)
            df = df.drop(columns=["Aliases_manual"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"Wrote: {OUT_PATH.resolve()}")
    print("Open the CSV and spot-check a few rows. Add manual_aliases.csv to tweak tricky brands (nicknames, Korean, etc.).")


if __name__ == "__main__":
    main()
