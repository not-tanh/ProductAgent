import os
import argparse
import sys
import traceback

import polars as pl


def process_cleaning(input_path: str, output_path: str, rejected_path: str):
    """
    Function to clean the Amazon Canada Products dataset.
    Splits data into 'clean' (for processing) and 'rejected' (for auditing).
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

    # 2. Define Validation Logic (Flagging Phase)
    # We create a boolean expression to identify valid rows without filtering immediately.
    is_valid_expression = (
        pl.col("asin").is_not_null() &
        pl.col("title").is_not_null() &
        (pl.col("title").str.len_chars() > 1) &
        pl.col("price").is_not_null() &
        (pl.col("price") > 0) &
        pl.col("imgUrl").is_not_null() &
        pl.col("imgUrl").str.contains(url_pattern) &
        pl.col("productURL").is_not_null() &
        pl.col("productURL").str.contains(url_pattern)
    )

    # Apply the validation flag
    df = df.with_columns(is_valid_expression.alias("is_valid"))

    # 3. Split Data (Separation Phase)
    # Rows that passed validation
    df_clean = df.filter(pl.col("is_valid")).drop("is_valid")

    # Rows that failed validation
    df_rejected = df.filter(~pl.col("is_valid")).drop("is_valid")

    # 4. Handle Rejected Data
    rejected_count = df_rejected.height
    if rejected_count > 0:
        print(f"[WARN] Found {rejected_count} invalid rows. Saving to {rejected_path}...")
        try:
            # Create directory for rejected file if needed
            rej_dir = os.path.dirname(rejected_path)
            if rej_dir and not os.path.exists(rej_dir):
                os.makedirs(rej_dir)

            # Save rejected data as CSV (easier for humans to inspect/debug)
            df_rejected.write_csv(rejected_path)
        except Exception as e:
            print(f"[ERROR] Could not save rejected file: {e}")
            traceback.print_exc()

    else:
        print("[INFO] No invalid rows found. Perfect data!")

    # 5. Process Clean Data (Deduplication & Enrichment)
    print(f"[INFO] Processing {df_clean.height} valid rows...")

    # Deduplication (Keep first occurrence of ASIN)
    df_clean = df_clean.unique(subset=["asin"], keep="first")

    # Imputation
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
            pl.lit(" | Title: ") + pl.col("title")
        ).alias("text_for_embedding")
    )

    # 6. Final Statistics
    final_count = df_clean.height
    total_removed = original_count - final_count

    print(f"[STATS] ------------------------------------------------")
    print(f"[STATS] Original Rows  : {original_count}")
    print(f"[STATS] Invalid Rows   : {rejected_count} (Saved to rejected file)")
    print(f"[STATS] Clean Rows     : {final_count} (Ready for embedding)")
    print(f"[STATS] Total Removed  : {total_removed} (Invalid + Duplicates)")
    print(f"[STATS] ------------------------------------------------")

    # 7. Save Clean Data
    print(f"[INFO] Saving clean data to {output_path}...")
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        df_clean.write_parquet(output_path)
        print(f"[INFO] Success! Process complete.")
    except Exception as e:
        print(f"[ERROR] Failed to save clean file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Script: Clean Amazon Dataset & Track Rejected Rows.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Path to raw CSV input")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to processed Parquet output")

    # New argument for the rejected file
    parser.add_argument("-r", "--rejected", type=str, default="data/rejected_rows.csv", help="Path to save invalid rows (default: data/rejected_rows.csv)")

    args = parser.parse_args()

    process_cleaning(args.input, args.output, args.rejected)
