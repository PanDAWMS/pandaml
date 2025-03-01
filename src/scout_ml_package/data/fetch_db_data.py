# # # data/fetch_db_data.py

import warnings
import configparser
import os
import oracledb
import pandas as pd
from typing import List

warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

oracledb.init_oracle_client(
    config_dir="/data/model-data/configs",
    lib_dir="/opt/oracle/instantclient/instantclient_19_25",
)


class DatabaseFetcher:
    """
    Fetches data from a database using a specified configuration.

    Attributes:
    - db_config_name (str): Name of the database configuration section.
    - config (configparser.ConfigParser): Loaded configuration.
    - conn (oracledb.Connection): Database connection.
    """

    def __init__(self, db_config_name: str):
        """
        Initializes the DatabaseFetcher with a database configuration name.

        Parameters:
        - db_config_name (str): Name of the database configuration section.

        Raises:
        - FileNotFoundError: If the configuration file does not exist.
        - KeyError: If the specified section is missing in the configuration file.
        """
        self.db_config_name = db_config_name
        self.config = self.load_config()
        self.conn = self.get_db_connection()

    def load_config(self) -> configparser.ConfigParser:
        """
        Loads the configuration from a file.

        Returns:
        - configparser.ConfigParser: Loaded configuration.

        Raises:
        - FileNotFoundError: If the configuration file does not exist.
        - KeyError: If the specified section is missing in the configuration file.
        """
        config = configparser.ConfigParser()

        # Get the directory of the current script and construct path to config.ini
        config_path = os.path.join("/data/model-data/configs", "", "config.ini")

        # Check if the configuration file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # Read configuration file
        config.read(config_path)

        # Check if the specified section exists in the config file
        if self.db_config_name not in config:
            raise KeyError(f"'{self.db_config_name}' section missing in config file")

        return config

    def get_db_connection(self) -> oracledb.Connection:
        """
        Establishes a connection to the database.

        Returns:
        - oracledb.Connection: Database connection.

        Raises:
        - Exception: If connection fails.
        """
        # Get database credentials
        database_user = self.config[self.db_config_name].get("user")
        database_password = self.config[self.db_config_name].get("password")
        dsn = self.config[self.db_config_name].get("dsn")

        try:
            # Establish connection with tcp_connect_timeout
            conn = oracledb.connect(
                user=database_user,
                password=database_password,
                dsn=dsn,
                tcp_connect_timeout=20,
            )  # 20 seconds timeout
            return conn

        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            return None

    def reconnect_if_needed(self) -> None:
        """
        Reconnects to the database if the connection is not healthy.
        """
        if not self.conn or not self.conn.is_healthy():
            self.close_connection()
            self.conn = self.get_db_connection()
            if not self.conn:
                raise Exception("Failed to reconnect to the database.")

    def fetch_task_param(self, jeditaskids: List[int]) -> pd.DataFrame:
        """
        Fetches task parameters for the specified JEDITASKIDs.

        Parameters:
        - jeditaskids (List[int]): List of JEDITASKIDs.

        Returns:
        - pd.DataFrame: DataFrame containing task parameters.
        """
        self.reconnect_if_needed()

        if not isinstance(jeditaskids, list):
            jeditaskids = [jeditaskids]

        jeditaskid_str = ", ".join(map(str, jeditaskids))

        # Combined SQL query
        query = f"""
        SELECT
            jt.jeditaskid,
            jt.prodsourcelabel,
            jt.processingtype,
            jt.transhome,
            jt.transpath,
            jt.cputimeunit,
            jt.corecount,
            SUM(jd.NFILES) AS total_nfiles,
            SUM(jd.NEVENTS) AS total_nevents,
            COUNT(jd.DATASETNAME) AS distinct_datasetname_count
        FROM
            atlas_panda.jedi_tasks jt
        LEFT JOIN
            atlas_panda.jedi_datasets jd ON jt.jeditaskid = jd.jeditaskid
        WHERE
            jt.jeditaskid IN ({jeditaskid_str})  and (jd.type = 'input' or jd.type = 'pseudo_input')
        GROUP BY
            jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.taskname, jt.corecount
        """
        df = pd.read_sql(query, con=self.conn)
        return df

    def write_data(self, data: pd.DataFrame, table_name: str, max_retries: int = 3) -> None:
        """
        Writes a pandas DataFrame to an Oracle database table with retry mechanism.

        Parameters:
        - data (pd.DataFrame): DataFrame containing data to be inserted.
        - table_name (str): Name of the target database table.
        - max_retries (int): Maximum number of retries if connection fails. Defaults to 3.
        """
        retries = 0
        while retries <= max_retries:
            self.reconnect_if_needed()  # Ensure connection is active

            if not self.conn:
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Failed to connect to the database after {max_retries} retries.")
                else:
                    print(f"Connection not active. Retrying...")
                    continue

            try:
                # Generate SQL placeholders for insertion
                columns = ", ".join(data.columns)
                placeholders = ", ".join([":" + str(i + 1) for i in range(len(data.columns))])
                sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

                # Convert DataFrame rows to a list of tuples
                rows = [tuple(row) for row in data.itertuples(index=False)]

                # Execute batch insertion using a with statement for the cursor
                with self.conn.cursor() as cursor:
                    cursor.executemany(sql, rows)
                    self.conn.commit()

                print(f"Data successfully written to {table_name}")
                break
            except Exception as e:
                print(f"Failed to write data to {table_name}: {e}")
                self.conn.rollback()  # Rollback in case of error
                retries += 1
                if retries > max_retries:
                    raise Exception(f"Failed to write data after {max_retries} retries.")
                else:
                    print(f"Retrying...")

    def get_connection(self) -> oracledb.Connection:
        """
        Returns the current database connection.

        Returns:
        - oracledb.Connection: Database connection.
        """
        self.reconnect_if_needed()
        return self.conn

    def close_connection(self) -> None:
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            print("Database connection closed successfully.")
            self.conn = None
        else:
            print("No active database connection to close.")
