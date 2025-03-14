import os
import pandas as pd
import mysql.connector
from google.cloud import storage
import logging
import io


def get_df_from_storage(bucket_name, file_name):
    """
    Download a file from Cloud Storage to a temporary location.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob_data = io.BytesIO()
    blob.download_to_file(blob_data)

    blob_data.seek(0)

    df = pd.read_csv(blob_data)
    return df

def get_db_connection():
    """
    Create a SQLAlchemy database connection
    Uses Cloud SQL connection via Unix socket for Cloud Run
    """
    # Get environment variables with defaults for testing
    # db_user = os.environ.get('DB_USER', 'varun')
    db_user = 'varun'
    db_pass = """;F6y#"9DCv%<h6?>"""
    db_name = 'combined_transaction_data'
    cloud_sql_connection_name = 'primordial-veld-450618-n4:us-central1:mlops-sql'
    
    # Print for debugging
    print(f"DB_USER: {db_user}")
    print(f"DB_NAME: {db_name}")
    print(f"CLOUD_SQL_CONNECTION_NAME: {cloud_sql_connection_name}")
    
    # Construct connection string directly
    socket_dir ='/cloudsql'
    socket_path = f"{socket_dir}/{cloud_sql_connection_name}"
    
    # Create connection string
    conn_str = f"mysql+pymysql://{db_user}:{db_pass}@/{db_name}?unix_socket={socket_path}"
    
    # Print masked connection string for debugging
    masked_conn_str = conn_str.replace(db_pass, "******")
    print(f"Connection string: {masked_conn_str}")
    
    # Create SQLAlchemy engine
    return sqlalchemy.create_engine(conn_str)

def process_csv_to_mysql(bucket_name, file_name):
    """
    Process CSV file from GCS and load into MySQL
    """
    df = get_df_from_storage(bucket_name, file_name)
    
    # Replace NaN values
    df = df.fillna({
        'lag_1': 0,
        'lag_7': 0,
        'rolling_mean_7': 0.0
    })
    
    # Ensure correct data types
    df['date'] = pd.to_datetime(df['date'])
    
    # Database connection
    engine = get_db_connection()

    print("++++++++++++++++")
    print(engine)

    return True
    
    try:
        # Insert into PRODUCT table (if not exists)
        product_df = df[['Product Name']].drop_duplicates()
        product_df.to_sql(
            'PRODUCT', 
            engine, 
            if_exists='ignore', 
            index=False, 
            dtype={
                'product_name': sqlalchemy.types.VARCHAR(255)
            }
        )
        
        # Insert into TIME_DIMENSION table (if not exists)
        time_dim_df = df[['date', 'day_of_week', 'is_weekend', 'day_of_month', 'day_of_year', 'month', 'week_of_year']].drop_duplicates()
        time_dim_df.to_sql(
            'TIME_DIMENSION', 
            engine, 
            if_exists='ignore', 
            index=False, 
            dtype={
                'date': sqlalchemy.types.DATE(),
                'day_of_week': sqlalchemy.types.INTEGER(),
                'is_weekend': sqlalchemy.types.BOOLEAN(),
                'day_of_month': sqlalchemy.types.INTEGER(),
                'day_of_year': sqlalchemy.types.INTEGER(),
                'month': sqlalchemy.types.INTEGER(),
                'week_of_year': sqlalchemy.types.INTEGER()
            }
        )
        
        # Insert into SALES table
        df.to_sql(
            'SALES', 
            engine, 
            if_exists='append', 
            index=False, 
            dtype={
                'date': sqlalchemy.types.DATE(),
                'product_name': sqlalchemy.types.VARCHAR(255),
                'total_quantity': sqlalchemy.types.INTEGER(),
                'lag_1': sqlalchemy.types.INTEGER(),
                'lag_7': sqlalchemy.types.INTEGER(),
                'rolling_mean_7': sqlalchemy.types.FLOAT()
            }
        )
        
        print(f"Successfully processed {file_name}")
        return True
    
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return False


def insert_on_conflict_update_product(table, conn, keys, data_iter):
    """Insert into SALES and update on conflict"""
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(data)

    stmt = stmt.on_duplicate_key_update(
        product_name=stmt.inserted.product_name
    )

    result = conn.execute(stmt)
    return result.rowcount

def insert_on_conflict_update_time_dimension(table, conn, keys, data_iter):
    """Insert into TIME_DIMENSION and update on conflict"""
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(data)
    
    stmt = stmt.on_duplicate_key_update(
        day_of_week=stmt.inserted.day_of_week,
        is_weekend=stmt.inserted.is_weekend,
        day_of_month=stmt.inserted.day_of_month,
        day_of_year=stmt.inserted.day_of_year,
        month=stmt.inserted.month,
        week_of_year=stmt.inserted.week_of_year
    )

    result = conn.execute(stmt)
    return result.rowcount

def insert_on_conflict_update_sales(table, conn, keys, data_iter):
    """Insert into SALES and update on conflict"""
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(data)

    stmt = stmt.on_duplicate_key_update(
        total_quantity=stmt.inserted.total_quantity
    )

    result = conn.execute(stmt)
    return result.rowcount