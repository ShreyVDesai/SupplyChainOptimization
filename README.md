# Supply Chain Data Pipeline

This project implements an automated data pipeline for supply chain data processing. It watches for new files in a Google Cloud Storage bucket, validates them, and processes them to prepare data for analysis.

## Architecture

The pipeline consists of the following components:

1. **Airflow DAG** - Orchestrates the entire pipeline flow
2. **Data Validation** - Checks schema and data quality
3. **Data Preprocessing** - Cleans and transforms raw data
4. **Google Cloud Storage Integration** - For data source and destination

## Setup Instructions

### Prerequisites

- Python 3.8+
- Apache Airflow 2.0+
- Google Cloud Storage account and bucket
- Docker (optional, for containerized deployment)

### Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd SupplyChainOptimization
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up Google Cloud credentials:

   - Place your GCP service account key file in `secret/gcp-key.json`
   - Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

4. Configure Airflow:
   - Add the DAG directory to your Airflow DAGs folder
   - Create a Google Cloud connection in Airflow named `google_cloud_default`

### Configuration

The main configuration parameters are set in `dags/data_pipeline_dag.py`:

- `BUCKET_NAME`: The GCS bucket to monitor for new files
- `TEMP_LOCAL_PATH`: Local directory for temporary files
- `PROCESSED_PREFIX`: Prefix for processed files to avoid reprocessing

## How It Works

1. The DAG monitors the GCS bucket for any new files
2. When a new file is detected, it's downloaded for processing
3. The validation script checks for schema changes and data quality issues
4. If validation passes (or issues are accepted), the preprocessing script cleans the data
5. The cleaned data is stored back in GCS for further analysis

## Testing Locally

You can test the pipeline locally using the provided test script:

```bash
cd Data-Pipeline/scripts
python test_pipeline.py path/to/input_file.csv path/to/output_file.csv
```

This will run the validation and preprocessing steps without requiring Airflow.

## Troubleshooting

### Common Issues

1. **DAG not triggering**:

   - Verify that your GCS bucket permissions are correct
   - Check that Airflow's Google Cloud connection is configured properly
   - Ensure the DAG is enabled in the Airflow UI

2. **Validation errors**:

   - Review the logs to identify the specific schema or data quality issue
   - If schema changes are expected, you may need to update the reference schema

3. **Preprocessing failures**:
   - Check input file format and contents
   - Ensure all required Python packages are installed
   - Review preprocessing logs for specific error messages

### Logs

- Airflow logs can be viewed in the Airflow UI
- Local test logs are saved to `pipeline_test.log`
- Individual script logs are available in their respective directories

## File Structure

```
SupplyChainOptimization/
├── Data-Pipeline/
│   ├── scripts/
│   │   ├── DataValidation_Schema&Stats.py  # Schema validation
│   │   ├── dataPreprocessing.py            # Data cleaning and transformation
│   │   ├── test_pipeline.py                # Local testing script
│   │   └── logger.py                       # Logging utilities
├── dags/
│   └── data_pipeline_dag.py                # Airflow DAG definition
├── secret/
│   └── gcp-key.json                        # GCP service account key (not in repo)
└── README.md                               # This documentation
```

## Development

To contribute to this project:

1. Create a new branch for your feature or fix
2. Make your changes with appropriate tests
3. Ensure all tests pass
4. Submit a pull request with a clear description of your changes

## License

[Include license information]
