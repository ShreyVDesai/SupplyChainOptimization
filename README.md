# Supply Chain Optimization

This project contains a supply chain optimization system with Airflow-based data pipelines.

## Environment Setup

### Quick Start

1. Ensure you have Docker and Docker Compose installed
2. Place your Google Cloud Platform key file at `./secret/gcp-key.json`
3. Configure the `.env` file in the root directory with your environment variables
4. Start the services with a simple command:
   ```
   docker compose up -d
   ```
5. Access the Airflow UI at http://localhost:8080 (default credentials: admin/admin)
6. To stop all services:
   ```
   docker compose down
   ```

### Notes

- The project uses a single `.env` file for all configuration
- All environment variables are loaded directly by docker-compose
- The GCP key file should be placed in the secret directory (and is gitignored)

## Environment Configuration

### Required Environment Variables

The .env file includes all necessary variables:

- `AIRFLOW_UID`: User ID for Airflow containers (default: 50000)
- `AIRFLOW_FERNET_KEY`: Encryption key for sensitive data
- `AIRFLOW_DATABASE_PASSWORD`: Password for Airflow database
- Various admin user credentials and database settings

### GCP Authentication

GCP credentials are stored in:

- `secret/gcp-key.json`: Your GCP service account key

## Accessing Airflow

- Web UI: http://localhost:8080
- Default login: admin/admin
