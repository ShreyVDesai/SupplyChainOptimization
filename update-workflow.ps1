$workflowContent = @"
name: Data-Pipeline

on:
  push:
    branches:
      - data-pipeline-deployment

jobs:
  test-and-build:
    name: Test and Build
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r Data_Pipeline/requirements.txt

      - name: Set PYTHONPATH
        run: echo "PYTHONPATH=\$PYTHONPATH:\$(pwd)/Data_Pipeline" >> \$GITHUB_ENV

      - name: Run Unit Tests
        run: python -m unittest discover -s tests -p "test*.py"

      - name: Set Up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Load Environment Variables
        run: |
          echo "AIRFLOW_UID=50000" >> \$GITHUB_ENV
          echo "DOCKER_GID=1000" >> \$GITHUB_ENV

      - name: Start Docker Compose
        run: docker-compose up -d --build

      - name: Verify Containers are Running
        run: docker ps -a
"@

Set-Content -Path ".github/workflows/data-pipeline.yml" -Value $workflowContent 