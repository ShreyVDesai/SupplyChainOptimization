# Supply Chain Optimization Import Structure

This document explains the import structure of the Supply Chain Optimization project and how to use it.

## Import Structure

The project supports two types of imports:

1. **Absolute imports** (used in Airflow DAGs):

   ```python
   from Data_Pipeline.scripts.logger import logger
   from Data_Pipeline.scripts.utils import send_email
   ```

2. **Relative imports** (used in unit tests and local development):
   ```python
   from logger import logger
   from utils import send_email
   ```

All modules in the `Data_Pipeline/scripts` directory are designed to work with both import styles automatically.

## How It Works

Each module uses a try-except pattern to handle both import scenarios:

```python
try:
    # First try local import
    from logger import logger
except ImportError:
    # Fall back to absolute import if local fails
    from Data_Pipeline.scripts.logger import logger
```

## Usage in Different Environments

### In Airflow DAGs

When running in Airflow, use absolute imports because the workspace root (`/opt/airflow/`) is different from the script directory (`/app/scripts/`):

```python
from Data_Pipeline.scripts.preprocessing import main as preprocessing_main
```

### In Unit Tests

When running unit tests, you can use relative imports because the tests run from within the `Data_Pipeline/tests` directory, which has access to the `scripts` directory:

```python
from scripts.utils import send_email
```

### For Local Development

For local development, you can use either style depending on your PYTHONPATH setup:

1. If your PYTHONPATH includes the project root, use absolute imports:

   ```python
   from Data_Pipeline.scripts.utils import send_email
   ```

2. If you're running scripts directly from within the `scripts` directory, use relative imports:
   ```python
   from utils import send_email
   ```

## Docker Environment

In the Docker container, both import styles work because:

1. The Docker container's `WORKDIR` is set to `/app/scripts`, making relative imports work
2. The Python package is installed, making absolute imports work

## Installation

To install the package for development:

```bash
pip install -e .
```

This will install the package in development mode, allowing both absolute and relative imports to work.
