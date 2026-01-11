import argparse

import duckdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()

    conn = duckdb.connect(f'{args.output}')

    conn.execute(f"""
        CREATE OR REPLACE TABLE products AS 
        SELECT * FROM read_parquet('{args.input}')
    """)

    conn.execute("CREATE INDEX idx_category ON products(categoryName)")
    conn.execute("CREATE INDEX idx_rating ON products(stars)")
    conn.execute("CREATE INDEX idx_bought ON products(boughtInLastMonth)")
    conn.execute("CREATE INDEX idx_bestseller ON products(isBestSeller)")
    conn.execute("CREATE INDEX idx_price ON products(price)")
    conn.execute("CREATE INDEX idx_reviews ON products(reviews)")

    conn.close()
