# Acquire data from Kaggle
kaggle datasets download -d asaniczka/amazon-canada-products-2023-2-1m-products -p data/raw --unzip

# Clean & validate data
python data_pipeline/cleaning.py -i data/raw/amz_ca_total_products_data_processed.csv -o products.parquet

# Create duckdb for analysis
python data_pipeline/create_duckdb.py -i products.parquet -o products.duckdb

# Ingest into Qdrant
python data_pipeline/ingest_job.py -i products.parquet
