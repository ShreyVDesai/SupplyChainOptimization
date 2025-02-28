# Supply Chain Optimization

This project contains a supply chain optimization system with Airflow-based data pipelines.

## Quick Start

### Starting the Services

You can start the services in two ways:

1. **Using Docker Compose directly**:

   ```bash
   docker compose up
   ```

2. **Using the start script** (advanced/custom configurations):
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

- Default configuration is in the root `.env` file
- Custom secrets should be stored in `secret/.env`
- GCP credentials should be in `secret/gcp-key.json`

## Accessing Airflow

- Web UI: http://localhost:8080
- Default login: admin/admin
