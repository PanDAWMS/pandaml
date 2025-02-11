import oracledb
import pandas as pd
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
import sqlite3
oracledb.init_oracle_client(config_dir='/data/model-data/configs', lib_dir="/opt/oracle/instantclient/instantclient_19_25")

base_path = "/data/model-data/" 
# Create instances for input and output databases
input_db = DatabaseFetcher('database')
output_db = DatabaseFetcher('output_database')

# Check if connections are loaded successfully
input_connection_status = input_db.conn is not None
output_connection_status = output_db.conn is not None

print(f"Input database connection status: {input_connection_status}")
print(f"Output database connection status: {output_connection_status}")

sample_tasks = [27766704, 27746332]
r = input_db.fetch_task_param(sample_tasks)

print(r)

query = """
SELECT MAX(JediTaskID) FROM ATLAS_PANDA.PANDAMLTEST
"""
df = pd.read_sql(query, con=output_db.get_connection())
print(df)

# # Create a sample DataFrame to write
# sample_data = pd.DataFrame({
#     'column1': [1, 2, 3],
#     'column2': ['a', 'b', 'c']
# })

# # Write the sample data to the output database
# output_db.write_data(sample_data, 'output_table_name')
