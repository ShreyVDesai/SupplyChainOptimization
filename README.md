# Supply Chain Optimization

A comprehensive system for demand forecasting.

## Project Overview

### Data Pipeline

The data pipeline is designed to process, clean, and analyze supply chain data, specifically focusing on raw transaction data. It implements MLOps best practices including data versioning, workflow orchestration and testing.

### Key Features

- **Data Generation**: Realistic sample data generation incoprating trends from commodity ETFs from Yahoo Finance and synthetic transaction data
- **Data Pipeline**: Robust data processing pipeline using Apache Airflow and data versioning using DVC
- **Data Quality**: Validation, cleaning, and testing of data integrity
- **Containerization**: Docker-based deployment for consistent environments
- **Cloud Integration**: Google Cloud Platform for storage and processing

## Architecture

The project consists of several key components:

1. **Data Generation Layer**: Scripts that generate synthetic data based on real-world patterns

   - Generates synthetic demand data from Yahoo Finance commodity ETFs
   - Creates transaction data modeling store-customer interactions
   - Introduces realistic data quality issues for testing purposes

2. **Data Pipeline Layer**: Apache Airflow DAGs that orchestrate data movement and processing

   - Data ingestion, cleaning, validation, and transformation
   - Schema validation and anomaly detection using Great Expectations
   - Processing logging

3. **Storage Layer**: Various storage solutions for data
   - Local file storage for development
   - DVC for data versioning
   - Google Cloud Storage for production data

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ShreyVDesai/SupplyChainOptimization.git
cd SupplyChainOptimization
```

2. Create and configure environment variables:

Contact team for .env files, secrets and instructions to set this up.

3. Start the containers:

```bash
docker-compose up -d
```

4. Access Airflow UI at http://localhost:8080 (default credentials: admin/admin)

### Development Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Install additional dependencies for the data pipeline:

```bash
pip install -r requirements.txt
```

## Data Components

### Data Generation

The project includes scripts to generate synthetic data:

1. **Demand Data Generator (`dataGenerator.py`)**:

   - Uses Yahoo Finance data for realistic commodity price patterns
   - Applies daily, weekly, and seasonal variations
   - Incorporates economic factors like price elasticity

2. **Transaction Generator (`transactionGenerator.py`)**:

   - Creates realistic transaction records based on demand data
   - Models customer purchasing patterns
   - Simulates pricing with inflation adjustment over time

3. **Data Quality Simulation (`dataMess.py`)**:
   - Introduces typical data quality issues for testing
   - Creates missing values, duplicates, and inconsistent formats
   - Simulates real-world data collection problems

### Data Pipeline

The data pipeline performs the following steps:

1. **Data Acquisition**: Frontend will upload data onto the gcp bucket. It is simulated right now with a manual upload
2. **Data Preprocessing**: Cleaning, aggregation, and transformation
3. **Data Validation**: Schema validation and anomaly detection
4. **Data Versioning**: Using DVC to track data changes
5. **Feature Engineering**: Creating useful features for analysis
6. **Output Generation**: Creating clean datasets for analysis

## Bias Justification

This project is designed to be free from demographic biases for the following reasons:

1. **Data Source Selection**: The primary data sources (commodity ETFs, price data) are economic indicators not tied to demographics
2. **Transaction Anonymization**: The transaction data is randomly generated without demographic attributes
3. **Store Location Neutrality**: Store locations are generic names without geographic or demographic correlations
4. **Focus on Supply Chain**: The project analyzes commodity movement and time-focussed demand rather than consumer behavior
5. **No Personal Identifiers**: The system avoids collecting or using personal information that could introduce bias

This design explicitly addresses potential bias concerns by:

- Not collecting demographic data such as race, gender, age, etc.
- Using random transaction IDs instead of personal identifiers
- Focusing on commodity data that is not demographically influenced
- Generating synthetic data with statistical distributions unrelated to demographic factors

## How this submission meets assignment requirements

1. **Proper Documentation**: The README provides a comprehensive overview of the project, setup instructions and the project's design information. The code is commented and provides the logical emplanation of each script.
2. **Modular Syntax and Code**: The code is written in a modular format, ensuring that each file is for a specific purpose in the pipeline. Some code is abstracted out into the utils.py file.
3. **Pipeline Orchestration**: We are using AIrflow DAGs for pipeline orchestration. Code contains try-except blocks for error handling. The errors will be handled more gracefully upon frontend implementation.
4. **Tracking and Logging**: We are using python and Airflow's logger for tracking and logging. Alerts are sent in form of emails for now which will be changed upon frontend implementation.
5. **Data Version Control (DVC)**: We have implemented DVC for data versioning.
6. **Pipeline Flow Optimization**: We are using Airflow's gantt charts to identify bottlenecks in the pipeline.
7. **Schema and Statistics Generation**: We are using Great expectations to automatically generate data schema and statistics. This happens once for the processed data.
8. **Anomalies Detection and Alert Generation**: We are using the IQR to identify records that fall outside 3 standard deviations and flagging them as anomalies. The preprocessing script handles issues such as missing values and outliers and validation scripts handles schema violations. Alerts are sent in form of emails.
9. **Bias Detection and Mitigation**: We are not doing this as the nature of our data doesn't contain any personal data or otherwise that can introduce bias. Please refer to the bias justification point above.
10. **Test Modules**: We have a testing coverage of 95%+.
11. **Reproducibility**: The pipeline is reproducable and clear instructions for reprodicibility have been added to the README.
12. **Error Handling and Logging**: Error handling mechanisms have been included and logs contain enough information for easy troubleshooting.

## Configuration

The project uses several configuration files:

- `.env`: Environment variables for Docker and services
- `docker-compose.yaml`: Container configuration
- `Data_Pipeline/requirements.txt`: Python dependencies

### Google Cloud Configuration

To set up Google Cloud integration:

1. Create a service account with appropriate permissions
2. Download the key file and save it in the `secret/` directory as `gcp-key.json`
3. Update the `.env` file with your Google Cloud project information

## Monitoring and Logging

- Airflow provides task-level monitoring through its UI
- Custom logs are stored in the `logs/` directory
- Performance metrics are available in the Airflow UI

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

- Yahoo Finance API for providing commodity data
- Apache Airflow for workflow orchestration
- Docker for containerization
