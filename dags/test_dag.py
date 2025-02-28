from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def print_hello():
    print("Hello from the test DAG!")
    return "Hello World!"


# Define the DAG
with DAG(
    "test_hello_world",
    default_args=default_args,
    description="A simple test DAG that prints Hello World",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["test"],
) as dag:

    hello_task = PythonOperator(
        task_id="hello_task",
        python_callable=print_hello,
    )
