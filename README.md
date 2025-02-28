# Supply Chain Optimization

This project contains a supply chain optimization system with Airflow-based data pipelines.

## Quick Start

### Starting the Services

You can start the services in two ways:

1. **Using Docker Compose directly** (for development):

   ```bash
   docker compose up
   ```

2. **Using the start script** (for production/custom configurations):
   ```bash
   ./start-airflow.sh
   ```

### Stopping the Services

1. **Using Docker Compose directly**:

   ```bash
   docker compose down
   ```

2. **Using the stop script**:
   ```bash
   ./stop-airflow.sh
   ```

## Environment Configuration

This project uses a dual .env file setup:

1. **Root `.env` file**

   - Used automatically by `docker compose up`
   - Contains safe default values for development
   - Good for local testing and development

2. **`secret/.env` file**
   - Used when running `./start-airflow.sh`
   - Should contain your actual production credentials
   - Overrides values from the root .env file
   - Protected by .gitignore (never committed to version control)

### Required Environment Variables

Both .env files include all necessary variables:

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
