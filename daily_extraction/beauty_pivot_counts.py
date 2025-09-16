"""This is a script for transforming a dataset by pivoting specific columns and saving the results to a file.

The script reads a CSV input file containing beauty_daily counts of branded mentions, aggregates the counts into a pivot
table, and saves the processed data into a new CSV file. This script checks if the input file is empty or missing
specific required columns, handling these cases gracefully. The resulting pivot table aggregates values by date
and keyword while filling any missing values with 0. The output file is written in the same directory as the
input file.

Functions
---------
- make_pivot: Creates a pivot table from a pandas DataFrame over specified value columns.

"""

import pandas as pd
from pathlib import Path
from datetime import date, timedelta

INPUT_PATH = Path('data/beauty_alias_brand_daily_counts.csv')
OUT_POSTS  = Path('data/beauty_alias_pivoted_brand_counts.csv')


def make_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.pivot_table(
            index='date',
            columns='keyword',
            values=value_col,
            aggfunc='sum',
            fill_value=0
        )
        .sort_index()
    )


def main():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8')
    if df.empty:
        print('Input file is empty. Nothing to pivot.')
        return

    # Ensure required columns exist
    if 'post_mentions' not in df.columns:
        df['post_mentions'] = 0

    # Build pivot
    pivot_posts = make_pivot(df, 'post_mentions')

    # Save
    pivot_posts.to_csv(OUT_POSTS, encoding='utf-8')

    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    if yesterday_str in pivot_posts.index.astype(str):
        today_df = pivot_posts.loc[[yesterday_str]]
        out_today = Path('data/beauty_daily/') / f'brand_counts_{yesterday_str}.csv'
        today_df.to_csv(out_today, encoding='utf-8')
        print('Done. Wrote:')
        print(f'  posts -> {OUT_POSTS.resolve()}')
        print(f'  yesterday -> {out_today.resolve()}')
    else:
        print('Done. Wrote:')
        print(f'  posts -> {OUT_POSTS.resolve()}')
        print(f'No data for yesterday ({yesterday_str}) found in pivot.')


if __name__ == '__main__':
    main()
