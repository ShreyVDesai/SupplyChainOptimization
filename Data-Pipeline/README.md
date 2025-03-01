
## Step 1 - Data Generation 

### Description
This script generates synthetic demand data for products influenced by commodity ETFs (Exchange-Traded Funds). It incorporates stock price fluctuations, weekly demand variations, and sudden shocks (especially on weekends) to simulate realistic demand patterns. The generated data can be used for further analysis or modeling in supply chain and financial prediction tasks.

### Features
- Fetches stock market data for commodity ETFs (e.g., Corn, Wheat, Soybeans) using the Yahoo Finance API (`yfinance`).
- Simulates base demand with optional fixed values for specific products.
- Adds random noise and weekly demand fluctuations.
- Applies stock price influence on demand based on the ETF's historical price data.
- Introduces sudden demand shocks on weekends to mimic external events.
- Allows for customization of key parameters like base demand range, shock probability, and stock influence strength.
- Saves the generated synthetic demand data to an Excel file for further analysis.

### Input & Output
- **Input:** No direct input files. The script automatically fetches stock price data for commodities from Yahoo Finance.
- **Output:** An Excel file (`commodity_demand_{start_date}_{end_date}.xlsx`) containing the synthetic demand data for each product in the `PRODUCTS_TO_ETFS` dictionary, saved locally.

### Setup:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have access to the internet for the Yahoo Finance API to fetch stock data.

3. Create a `data/` directory where the output Excel file will be saved, or modify the script to specify a different path.

### Execution:
1. To run the script and generate synthetic demand data, execute:
   ```bash
   python dataGenerator.py
   ```

2. After execution, the synthetic demand data will be saved to an Excel file in the `../../data/` folder.

---

### Code Structure
- **`generate_synthetic_data()`**: Main function for generating synthetic demand data based on stock prices, noise, and weekly fluctuations.
- **`visualize_data()`**: Function to plot the generated demand data for visualization.
- **`save_to_excel()`**: Saves the generated synthetic demand data to an Excel file.
- **`PRODUCTS_TO_ETFS`**: Dictionary mapping products to their relevant commodity ETFs.
- **`Hyperparameters`**: Configurable parameters like start date, end date, and various ranges for demand and noise.

### Python Files
- [dataGenerator.py](./scripts/dataGenerator.py)
- [requirements.txt](./requirements.txt)

## Step 2 - Data Transaction Generation

### Description
This script generates synthetic transaction data for a retail business selling multiple products. It factors in commodity demand, base prices, inflation adjustments, and transaction patterns to simulate realistic sales data. The generated transaction data can be used for further analysis or modeling in supply chain management, price prediction, or financial analysis tasks.

### Features
- Generates transaction data from commodity demand in an Excel file.
- Simulates base prices with fixed price support for specific products.
- Adjusts prices for inflation based on historical rates.
- Simulates daily transactions with quantities and unit prices.
- Randomly assigns store locations and producer IDs.
- Assigns realistic timestamps within a business day (08:00 AM to 08:00 PM).
- Saves the generated data to an Excel file for analysis.

### Input & Output
- **Input:** Excel file (`commodity_demand_20190103_20241231.xlsx`) with commodity demand data.
- **Output:** Excel file (`transactions_{start_date}_{end_date}.xlsx`) with generated transaction data.

### Setup:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have the input demand data file (`commodity_demand_20190103_20241231.xlsx`) in the correct directory.

3. Create a `transaction/` directory where the output Excel file will be saved, or modify the script to specify a different path.

### Execution:
1. To run the script and generate transaction data, execute:
   ```bash
   python transactionGenerator.py
   ```

2. After execution, the synthetic transaction data will be saved to an Excel file in the `../../transaction/` folder.


### Code Structure
- **`load_demand_data()`**: Loads the demand data from the input Excel file.
- **`generate_base_prices()`**: Generates base prices for products based on predefined ranges and fixed prices for specific products.
- **`adjust_price_for_inflation()`**: Adjusts product prices for inflation based on historical inflation rates.
- **`transactionGenerator()`**: Main function to generate synthetic transaction data based on demand data, base prices, inflation adjustments, and random transaction generation.
- **`save_transactions_to_excel()`**: Saves the generated transaction data to an Excel file.

### Python Files
- [transactionGenerator.py](./scripts/transactionGenerator.py)
- [requirements.txt](./requirements.txt)


## Step 3 - Data Transaction Mess Generation
### Description
This script introduces a series of intentional data issues to an existing dataset, making it suitable for testing data cleaning techniques. The dataset is altered with missing values, inconsistent formats, outliers, errors, duplicates, logical inconsistencies, and deleted consecutive rows. This helps simulate real-world messy data for practicing data wrangling and preprocessing tasks. 

### Features
- Loads a dataset from an Excel file.
- Introduces missing values across different columns.
- Applies inconsistent date formats.
- Duplicates records and misspells product names.
- Inserts negative or zero values in the "Cost Price" and "Quantity" columns.
- Introduces invalid store locations.
- Adds logical inconsistencies, e.g., zero cost prices with non-zero quantities.
- Splits the dataset into smaller chunks to simulate large-scale data processing.
- Saves the modified dataset in a new Excel file for further use.

### Input & Output
- **Input:** Excel file (`transactions_20190103_20241231.xlsx`) containing the transaction data.
- **Output:** Excel file (`messy_transactions_20190103_20241231.xlsx`) containing the modified transaction data.

### Setup:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the input file (`transactions_20190103_20241231.xlsx`) is available in the specified directory.

### Execution:
1. To run the script and generate a messy dataset, execute:
   ```bash
   python dataMess.py
   ```
2. After execution, the messy dataset will be saved to an Excel file in the specified directory.

### Code Structure
- **`dataMess()`**: Main function that performs data manipulation including introducing missing values, outliers, errors, duplicates, and logical inconsistencies.

### Python Files
- [dataMess.py](./scripts/dataMess.py)
- [requirements.txt](./requirements.txt)



## Step 4 - Data Preprocessing
### Description
This script processes raw transaction data from CSV/XLSX files, performing data cleaning, standardization, anomaly detection, and transformation before saving the cleaned dataset locally or uploading it to Google Cloud Storage (GCS).

### Features
- Loads data from local storage or Google Cloud Storage (GCS).
- Cleans and standardizes feature types (dates, text case, numeric values).
- Handles missing values in dates and unit prices.
- Removes duplicate and invalid records.
- Detects anomalies in pricing, quantity, and transaction times.
- Standardizes product names using fuzzy matching.
- Extracts time-based features for further analysis.
- Saves cleaned data locally or uploads it to GCS.

### Input & Output
- **Input:** Excel file: messy_transactions_{start_date}_{end_date}.xlsx (either locally or in GCS).

- **Output:** Cleaned dataset saved as cleaned_data.csv (either locally or in cleaned_data/cleanedData.csv in GCS).
### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Google Cloud Storage credentials:
   - Create a service account with `storage.admin` permissions.
   - Download the service account key as `gcp-key.json` and place it in the `secret/` folder.

3. Set the GCP credentials environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/secret/gcp-key.json"
   ```
### Execution

1. Place your raw Excel files (`messy_transactions_{start_date}_{end_date}.xlsx`) in the `data/` folder.
2. Run the preprocessing script:
   - *Without GCS*:
     ```bash
     python scripts/dataPreprocessing.py
     ```
   - *With GCS*:
     ```bash
     python scripts/dataPreprocessing.py --cloud
     ```

### Code Structure
- **`load_data()` / `load_bucket_data()`**: Loads data from local storage or GCS.
- **`convert_feature_types()`**: Converts columns to appropriate data types.
- **`standardize_product_name()` / `apply_fuzzy_correction()`**: Cleans and corrects product names.
- **`detect_anomalies()`**: Identifies outliers in pricing, quantity, and timestamps.
- **`filling_missing_dates()`**: Fills missing date values.
- **`filling_missing_cost_price()`**: Handles missing unit prices.
- **`remove_invalid_records()` / `remove_duplicate_records()`**: Cleans up bad records.
- **`aggregate_daily_products()`**: Aggregates sales data by day and product.
- **`extracting_time_series_and_lagged_features()`**: Adds time-based features.
- **`upload_df_to_gcs()`**: Uploads cleaned data to GCS.
- **`send_anomaly_alert()`**: Sends an email notification if anomalies are detected.

### Python Files
- [dataPreprocessing.py](./scripts/dataPreprocessing.py)
- [logger.py](./scripts/logger.py)
- [sendMail.py](./scripts/sendMail.py)
- [requirements.txt](./requirements.txt)


## Step 5 - Data Uploader

### Description
This script uploads cleaned Excel data (converted into JSON) to a Google Cloud Storage (GCS) bucket. It converts all Excel files in a specified folder into JSON format and uploads them to a GCS bucket. The bucket name is derived from the Excel file name.

### Features
- Reads Excel files from a local directory.
- Converts Excel data to JSON with ISO-formatted dates.
- Uploads the JSON data to GCP buckets.
- Handles authentication for GCP using a service account key.
- Prints a sample of the JSON data for verification.

### Input & Output
- **Input:** Excel files (.xlsx or .xls) in a specified folder.
- **Output:** JSON files are uploaded to GCP buckets. The bucket name is based on the Excel file name (without extension).


### Setup
1. Install dependencies (same as preprocessing step).
2. Ensure `gcp-key.json` is configured properly.

### Execution
1. Place cleaned Excel files in `data/`.
2. Run the uploader script:
   ```bash
   python scripts/dataUploader.py
   ```

### Code Structure
- **`excel_to_json(file_path)`**: Converts an Excel file to a JSON string with ISO-formatted dates.
- **`pretty_print_json(json_str, num_records=5)`**: Prints a sample of the JSON data to verify the conversion.
- **`upload_json_to_gcs(bucket_name, json_data, destination_blob_name)`**: Uploads the JSON data to a GCP bucket.
- **`process_all_excel_files_in_data_folder(data_folder)`**: Main function that processes all Excel files in the given folder and uploads them to GCP.

### Python Files
- [dataUploader.py](./scripts/dataUploader.py)


## Step 6 - Data Validation

### Description
This script validates transaction data using Great Expectations, checking for required columns and data types. It generates a validation report, including schema checks and statistical results, and sends anomaly alerts via email if necessary.

### Features
- Loads transaction data from a CSV or Excel file.
- Validates required columns (`product_id`, `user_id`, `transaction_date`, `quantity`).
- Checks data types (e.g., `quantity` should be an integer).
- Generates a validation report with results saved as JSON.
- Attempts to generate DataDocs for metadata documentation.
- Sends email alerts for detected anomalies (e.g., high demand based on `quantity`).

### Input & Output
- **Input:** CSV or Excel file containing transaction data (e.g., `transactions_20190103_20241231.xlsx`).
- **Output:** JSON file with validation results saved locally.

### Setup
1. Install dependencies (same as previous steps).
2. Ensure `gcp-key.json` is set up.

### Execution
1. Place transaction data files in `data/`.
2. Run the validation script:
   ```bash
   python scripts/dataValidation.py
   ```

### Code Structure

- **`setup_logging()`**: Configures logging for the application to track progress.
- **`send_email()`**: Sends an email with the validation results or alert message. Handles text, HTML, and DataFrame content types.
- **`fetch_file_from_gcp()`**: Fetches a file from a Google Cloud Storage bucket and saves it locally.
- **`load_data()`**: Loads data from a CSV or Excel file into a Pandas DataFrame.
- **`validate_data()`**: Validates the data using Great Expectations, checks for required columns and data types, and saves the validation report as a JSON file.
- **`send_anomaly_alert()`**: Sends an email alert if anomalies are detected (e.g., high demand for products).
- **`main()`**: Main function to execute the entire workflow—fetches files, loads data, validates, and sends alerts.

### Python Files
- [dataValidation.py](./scripts/dataValidation_Schema&Stats.py)
- [sendMail.py](./scripts/sendMail.py)
- [requirements.txt](./requirements.txt)

---
