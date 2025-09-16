import pandas as pd


def main(subreddit_count=30):
    df = pd.read_csv("data/beauty_alias_brand_daily_counts.csv")
    all_subs = (df['top_subreddits'].dropna().astype(str).str.split(";"))
    flat = pd.Series([s.strip() for subs in all_subs for s in subs if s.strip()])
    top = flat.value_counts().head(subreddit_count)
    top_names = top.index.tolist()

    df_top = pd.DataFrame({"subreddit": [f"r/{name}" for name in top_names]})

    df_top.to_csv("files/beauty_subreddits_30.csv", index=False)


if __name__ == "__main__":
    main(30)
