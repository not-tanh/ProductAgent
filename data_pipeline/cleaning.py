import os
import argparse
import sys
import traceback

import polars as pl


def process_cleaning(input_path: str, output_path: str, rejected_path: str):
    """
    Function to clean Amazon Product Dataset with detailed rejection reasons.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print(f"[INFO] Starting to read data from {input_path}...")

    # 1. Load data
    try:
        df = pl.read_csv(input_path, ignore_errors=True)
    except Exception as e:
        print(f"[ERROR] Error reading file: {e}")
        traceback.print_exc()
        sys.exit(1)

    original_count = df.height
    print(f"[STATS] Original data: {original_count} rows")

    # Define Regex pattern for valid URLs
    url_pattern = r"^https?://\S+$"

    # ---------------------------------------------------------
    # 1. DETAILED VALIDATION LOGIC
    # ---------------------------------------------------------

    df = df.with_columns(
        pl.when(pl.col("asin").is_null())
        .then(pl.lit("Missing ASIN"))

        .when(pl.col("title").is_null())
        .then(pl.lit("Missing Title"))

        .when(pl.col("title").str.len_chars() <= 1)
        .then(pl.lit("Title too short"))

        .when(pl.col("price").is_null() | (pl.col("price") <= 0))
        .then(pl.lit("Invalid Price (Zero or Null)"))

        .when(pl.col("imgUrl").is_null() | ~pl.col("imgUrl").str.contains(url_pattern))
        .then(pl.lit("Invalid Image URL"))

        .when(pl.col("productURL").is_null() | ~pl.col("productURL").str.contains(url_pattern))
        .then(pl.lit("Invalid Product URL"))

        .otherwise(pl.lit(None))
        .alias("rejection_reason")
    )

    # ---------------------------------------------------------
    # 2. SPLIT DATA
    # ---------------------------------------------------------
    df_rejected = df.filter(pl.col("rejection_reason").is_not_null())
    df_clean = df.filter(pl.col("rejection_reason").is_null()).drop("rejection_reason")

    # ---------------------------------------------------------
    # 3. SAVE REJECTED REPORT
    # ---------------------------------------------------------
    rejected_count = df_rejected.height
    if rejected_count > 0:
        print(f"[WARN] Found {rejected_count} invalid rows.")

        reason_stats = df_rejected.group_by("rejection_reason").len().sort("len", descending=True)
        print("[STATS] Rejection Reasons Breakdown:")
        print(reason_stats)

        try:
            rej_dir = os.path.dirname(rejected_path)
            if rej_dir and not os.path.exists(rej_dir):
                os.makedirs(rej_dir)

            print(f"[INFO] Saving rejection report to {rejected_path}...")
            df_rejected.write_csv(rejected_path)
        except Exception as e:
            print(f"[ERROR] Could not save rejected file: {e}")
            traceback.print_exc()

    else:
        print("[INFO] No invalid rows found. Perfect data!")

    # ---------------------------------------------------------
    # 4. PROCESS CLEAN DATA (Deduplication & Enrichment)
    # ---------------------------------------------------------
    clean_rows_before_dedup = df_clean.height
    df_clean = df_clean.unique(subset=["asin"], keep="first")
    dedup_count = clean_rows_before_dedup - df_clean.height

    # Imputation & Feature Engineering
    df_clean = df_clean.with_columns([
        pl.col("stars").fill_null(0.0),
        pl.col("reviews").fill_null(0),
        pl.col("boughtInLastMonth").fill_null(0),
        pl.col("categoryName").fill_null("General"),
        pl.col("isBestSeller").fill_null(False),
        pl.col("title").str.strip_chars(),
    ])

    # Feature Engineering for Embedding
    df_clean = df_clean.with_columns(
        (
            pl.lit("Category: ") + pl.col("categoryName") +
            pl.lit("\nTitle: ") + pl.col("title")
        ).alias("text_for_embedding")
    )

    # ---------------------------------------------------------
    # 5. FINAL REPORT & SAVE
    # ---------------------------------------------------------
    final_count = df_clean.height
    print(f"\n[SUMMARY REPORT] ========================================")
    print(f"1. Original Rows       : {original_count}")
    print(f"2. Invalid Rows        : {rejected_count} (Saved to {rejected_path})")
    print(f"3. Duplicate Rows      : {dedup_count} (Removed)")
    print(f"4. Final Clean Rows    : {final_count} (Saved to {output_path})")
    print(f"========================================================\n")

    print(f"[INFO] Saving clean data to {output_path}...")
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df_clean.write_parquet(output_path)
        print(f"[INFO] Pipeline completed successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to save clean file: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-r", "--rejected", type=str, default="data/rejected_rows.csv")
    args = parser.parse_args()

    process_cleaning(args.input, args.output, args.rejected)
