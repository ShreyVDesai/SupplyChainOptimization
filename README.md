# Supply Chain Optimization

A comprehensive MLOps system for supply chain data processing, analysis, and optimization using Airflow, Docker, and Google Cloud Platform.

## Project Overview

The Supply Chain Optimization project is a data pipeline designed to process, clean, and analyze supply chain data, specifically focusing on commodity demand and transaction data. It implements MLOps best practices including data versioning, workflow orchestration, testing, and monitoring.

### Key Features

- **Data Generation**: Realistic sample data generation from Yahoo Finance stock data and synthetic transaction data
- **Data Pipeline**: Robust data processing pipeline using Apache Airflow
- **Data Quality**: Validation, cleaning, and testing of data integrity
- **Containerization**: Docker-based deployment for consistent environments
- **Cloud Integration**: Google Cloud Platform for storage and processing

## Architecture

The project consists of several key components:

1. **Data Generation Layer**: Scripts that generate synthetic data based on real-world patterns

   - `dataGenerator.py`: Generates demand data from Yahoo Finance commodity ETFs
   - `transactionGenerator.py`: Creates transaction data modeling store-customer interactions
   - `dataMess.py`: Introduces realistic data quality issues for testing purposes

2. **Data Pipeline Layer**: Apache Airflow DAGs that orchestrate data movement and processing

   - Data ingestion, cleaning, validation, and transformation
   - Schema validation and anomaly detection
   - Processing monitoring and logging

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
git clone https://github.com/yourusername/SupplyChainOptimization.git
cd SupplyChainOptimization
```

2. Create and configure environment variables:

```bash
cp .env.example .env
# Edit .env with your settings
```

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
pip install -r Data-Pipeline/requirements.txt
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

1. **Data Acquisition**: Fetching data from sources
2. **Data Preprocessing**: Cleaning, normalization, and transformation
3. **Data Validation**: Schema validation and anomaly detection
4. **Data Versioning**: Using DVC to track data changes
5. **Feature Engineering**: Creating useful features for analysis
6. **Output Generation**: Creating clean datasets for analysis

## Bias Justification

This project is designed to be free from demographic biases for the following reasons:

1. **Data Source Selection**: The primary data sources (commodity ETFs, price data) are economic indicators not tied to demographics
2. **Transaction Anonymization**: The transaction data is randomly generated without demographic attributes
3. **Store Location Neutrality**: Store locations are generic names without geographic or demographic correlations
4. **Focus on Supply Chain**: The project analyzes commodity movement and pricing rather than consumer behavior
5. **No Personal Identifiers**: The system avoids collecting or using personal information that could introduce bias

This design explicitly addresses potential bias concerns by:

- Not collecting demographic data such as race, gender, age, etc.
- Using random transaction IDs instead of personal identifiers
- Focusing on commodity data that is not demographically influenced
- Generating synthetic data with statistical distributions unrelated to demographic factors

## Configuration

The project uses several configuration files:

- `.env`: Environment variables for Docker and services
- `docker-compose.yaml`: Container configuration
- `Data-Pipeline/requirements.txt`: Python dependencies

### Google Cloud Configuration

To set up Google Cloud integration:

1. Create a service account with appropriate permissions
2. Download the key file and save it in the `secret/` directory as `gcp-key.json`
3. Update the `.env` file with your Google Cloud project information

## Monitoring and Logging

- Airflow provides task-level monitoring through its UI
- Custom logs are stored in the `logs/` directory
- Performance metrics are available in the Airflow UI

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

- Yahoo Finance API for providing commodity data
- Apache Airflow for workflow orchestration
- Docker for containerization
