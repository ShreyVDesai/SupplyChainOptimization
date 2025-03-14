import functions_framework
import os
import pandas as pd
from google.cloud import storage
import sqlalchemy
import io
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy.dialects.mysql import insert
import pymysql
from sqlalchemy import text
import numpy as np

def interpolate_missing_dates_for_product(df, product_name, last_date_sql):
    # Filter the dataframe for the specific product
    product_df = df[df['Product Name'] == product_name].copy()
    
    # Ensure 'date' column is in datetime format
    product_df['date'] = pd.to_datetime(product_df['date'])
    
    # Get the first date in the product's df and the last date in the SQL
    first_date_df = product_df['date'].min()
    last_date_sql = pd.to_datetime(last_date_sql)
    
    # Create a complete date range between the two dates
    full_date_range = pd.date_range(start=last_date_sql + pd.Timedelta(days=1), end=first_date_df - pd.Timedelta(days=1))
    
    # Create a DataFrame for missing dates
    missing_dates = pd.DataFrame({'date': full_date_range})
    
    # Forward fill, backward fill, and average both methods for total_quantity
    missing_dates['forward_fill'] = missing_dates['date'].apply(
        lambda x: product_df.loc[product_df['date'] <= x, 'total_quantity'].ffill().iloc[-1] if not product_df[product_df['date'] <= x].empty else np.nan)
    
    missing_dates['backward_fill'] = missing_dates['date'].apply(
        lambda x: product_df.loc[product_df['date'] >= x, 'total_quantity'].bfill().iloc[0] if not product_df[product_df['date'] >= x].empty else np.nan)
    
    # Average of both forward and backward fills
    missing_dates['interpolated_quantity'] = missing_dates[['forward_fill', 'backward_fill']].mean(axis=1)
    
    # Add product name to the missing data
    missing_dates['Product Name'] = product_name
    
    return missing_dates[['date', 'Product Name', 'interpolated_quantity']]

def get_db_connection() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    db_user = 'varun'
    db_pass = """;F6y#"9DCv%<h6?>"""
    db_name = 'combined_transaction_data'
    instance_connection_name = 'primordial-veld-450618-n4:us-central1:mlops-sql'
    
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    connector = Connector(ip_type=ip_type, refresh_strategy="LAZY")

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


def upsert_df(df: pd.DataFrame, table_name: str, engine):
    """
    Inserts or updates rows in a MySQL table based on duplicate keys.
    If a record with the same primary key exists, it will be replaced with the new record.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to insert/update.
        table_name (str): The target table name.
        engine: SQLAlchemy engine.
    """
    # Convert DataFrame to a list of dictionaries (each dict represents a row)
    data = df.to_dict(orient='records')
    
    # Build dynamic column list and named placeholders
    columns = df.columns.tolist()
    col_names = ", ".join(columns)
    placeholders = ", ".join(":" + col for col in columns)
    
    # Build the update clause to update every column with its new value
    update_clause = ", ".join(f"{col} = VALUES({col})" for col in columns)
    
    # Construct the SQL query using ON DUPLICATE KEY UPDATE
    sql = text(
        f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    
    # Execute the query in a transactional scope
    with engine.begin() as conn:
        conn.execute(sql, data)

@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data

    bucket_name = data["bucket"]
    file_name = data["name"]
    
    # if not file_name.lower().endswith('.csv'):
    #     print(f"Skipping non-CSV file: {file_name}")
    #     return

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))

     # Replace NaN values
    df = df.fillna({
        'lag_1': 0,
        'lag_7': 0,
        'rolling_mean_7': 0.0
    })
    
    # Ensure correct data types
    df['date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Database connection
    engine = get_db_connection()

    try:

        with engine.connect() as connection:

            last_date_sql_query = "SELECT MAX(date) FROM Sales;"
            result = connection.execute(text(last_date_sql_query))
            last_date_sql = result.fetchone()[0]


            unique_products = df['Product Name'].unique()
            
            # Initialize an empty list to collect interpolated data
            interpolated_data_list = []
            
            for product_name in unique_products:
                missing_data = interpolate_missing_dates_for_product(df, product_name, last_date_sql)
                interpolated_data_list.append(missing_data)
            
            # Concatenate all interpolated data back together
            interpolated_data = pd.concat(interpolated_data_list, ignore_index=True)
            
            # Combine interpolated data with the existing data (if necessary)
            df = pd.concat([df, interpolated_data], ignore_index=True)

            # unique_products = df[['Product Name']].drop_duplicates()
            # product_data = [(product,) for product in unique_products['Product Name']]

            # insert_stmt = text("""
            # INSERT INTO PRODUCT (product_name)
            # VALUES (%s)
            # ON DUPLICATE KEY UPDATE product_name = VALUES(product_name);
            # """)
            
            # connection.execute(insert_stmt, product_data)

        # Insert into PRODUCT table (if not exists)
        product_df = df[['Product Name']].drop_duplicates()
        product_df.rename(columns={'Product Name': 'product_name'}, inplace=True)
        # product_df.to_sql(
        #     'Product', 
        #     connection, 
        #     if_exists='append', 
        #     index=False, 
        #     method=insert_on_conflict_update_product
        # )
        print(product_df.head(10))
        upsert_df(product_df, 'Product', engine)
        
        # # # Insert into TIME_DIMENSION table (if not exists)
        time_dim_df = df[['date', 'day_of_week', 'is_weekend', 'day_of_month', 'day_of_year', 'month', 'week_of_year']].drop_duplicates()
        # time_dim_df.to_sql(
        #     'Time_Dimension', 
        #     connection, 
        #     if_exists='append', 
        #     index=False,
        #     method=insert_on_conflict_update_time_dimension
        # )
        upsert_df(time_dim_df, 'Time_Dimension', engine)
        
        # # # Insert into SALES table
        sales_df = df[['date', 'Product Name', 'Total Quantity']].drop_duplicates()
        sales_df.rename(columns={'Product Name': 'product_name', 'Total Quantity': 'total_quantity',}, inplace=True)
        #     sales_df.to_sql(
        #         'Sales', 
        #         connection, 
        #         if_exists='append', 
        #         index=False, 
        #         method=insert_on_conflict_update_sales
        #     )
        upsert_df(sales_df, 'Sales', engine)
            
        print(f"Successfully processed {file_name}")
        # return True
        blob.delete()
    
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return False


# compare the last date of this sql to the first date of the file and do interpolation
# delete the csv file in the bucket
