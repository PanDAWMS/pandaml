import configparser
import os
import logging
import json
import time
from typing import Tuple
from queue import Queue
import stomp


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

#
# class ConfigLoader:
#     def __init__(self, config_path):
#         self.config_path = config_path
#         self.mb_server_host_port = None
#         self.queue_name = None
#         self.vhost = None
#         self.username = None
#         self.passcode = None
#         self.load_config()
#
#     def load_config(self):
#         config = configparser.ConfigParser()
#         config.read(self.config_path)
#
#         # Extract credentials and other settings
#         self.mb_server_host_port = (config["LPORT"]["host"], int(config["LPORT"]["port"]))
#         self.queue_name = config["LPORT"]["queue_name"]
#         self.vhost = config["LPORT"]["vhost"]
#         self.username = os.environ.get(
#             "STOMP_USERNAME", config["credentials"]["username"]
#         )
#         self.passcode = os.environ.get(
#             "STOMP_PASSWORD", config["credentials"]["passcode"]
#         )
#
#
# class MyListener(stomp.ConnectionListener):
#     def __init__(
#         self, task_id_queue, mb_server_host_port, vhost, username, passcode, queue_name
#     ):
#         self.task_id_queue = task_id_queue
#         self.mb_server_host_port = mb_server_host_port
#         self.vhost = vhost
#         self.username = username
#         self.passcode = passcode
#         self.queue_name = queue_name
#         self.conn = None
#         self.connect_to_stomp()
#
#     def on_connected(self, frame):
#         logging.info(f"Connected: {frame.headers} {frame.body}")
#
#     def on_error(self, frame):
#         logging.error(f"Received an error: {frame.body}")
#
#     def on_message(self, frame):
#         try:
#             msg = json.loads(frame.body)
#             now_ts = time.time()
#             if (
#                 msg["msg_type"] == "task_status"
#                 and msg["status"] == "defined"
#                 and now_ts - msg["timestamp"] <= 600
#             ):
#                 task_id = msg["taskid"]
#                 logging.info(f"Received task ID: {task_id}")
#                 self.task_id_queue.put(task_id)
#         except json.JSONDecodeError as e:
#             logging.error(f"Error decoding JSON: {e}")
#
#     def connect_to_stomp(self):
#         max_retries = 5
#         retries = 0
#         while retries < max_retries:
#             try:
#                 self.conn = stomp.Connection12(
#                     [self.mb_server_host_port], vhost=self.vhost
#                 )
#                 self.conn.set_listener("", self)
#                 self.conn.connect(self.username, self.passcode, wait=True)
#                 logging.info("Connected to STOMP server")
#
#                 # Subscribe to the queue
#                 self.conn.subscribe(destination=self.queue_name, id=1, ack="auto")
#                 logging.info(f"Subscribed to queue: {self.queue_name}")
#                 break
#             except Exception as e:
#                 logging.error(f"Error connecting to STOMP server: {e}. Retrying...")
#                 retries += 1
#                 time.sleep(1)  # Wait before retrying


class ConfigLoader:
    """
    Loads configuration from a file.

    Attributes:
    - config_path (str): Path to the configuration file.
    - mb_server_host_port (Tuple[str, int]): Message broker server host and port.
    - queue_name (str): Name of the queue to subscribe to.
    - vhost (str): Virtual host for the message broker.
    - username (str): Username for authentication.
    - passcode (str): Password for authentication.
    """

    def __init__(self, config_path: str):
        """
        Initializes the ConfigLoader with a configuration file path.

        Parameters:
        - config_path (str): Path to the configuration file.

        Raises:
        - FileNotFoundError: If the configuration file does not exist.
        """
        self.config_path = config_path
        self.mb_server_host_port = None
        self.queue_name = None
        self.vhost = None
        self.username = None
        self.passcode = None
        self.load_config()

    def load_config(self) -> None:
        """
        Loads configuration from the specified file.

        Raises:
        - configparser.Error: If there's an issue parsing the configuration file.
        """
        config = configparser.ConfigParser()
        config.read(self.config_path)

        # Extract credentials and other settings
        self.mb_server_host_port = (config["LPORT"]["host"], int(config["LPORT"]["port"]))
        self.queue_name = config["LPORT"]["queue_name"]
        self.vhost = config["LPORT"]["vhost"]
        self.username = os.environ.get(
            "STOMP_USERNAME", config["credentials"]["username"]
        )
        self.passcode = os.environ.get(
            "STOMP_PASSWORD", config["credentials"]["passcode"]
        )


class MyListener(stomp.ConnectionListener):
    """
    Listens for messages on a STOMP queue.

    Attributes:
    - task_id_queue (Queue): Queue to store received task IDs.
    - mb_server_host_port (Tuple[str, int]): Message broker server host and port.
    - vhost (str): Virtual host for the message broker.
    - username (str): Username for authentication.
    - passcode (str): Password for authentication.
    - queue_name (str): Name of the queue to subscribe to.
    - conn (stomp.Connection): Connection to the STOMP server.
    """

    def __init__(
        self,
        task_id_queue: Queue,
        mb_server_host_port: Tuple[str, int],
        vhost: str,
        username: str,
        passcode: str,
        queue_name: str,
    ):
        """
        Initializes the MyListener with connection details and a task ID queue.

        Parameters:
        - task_id_queue (Queue): Queue to store received task IDs.
        - mb_server_host_port (Tuple[str, int]): Message broker server host and port.
        - vhost (str): Virtual host for the message broker.
        - username (str): Username for authentication.
        - passcode (str): Password for authentication.
        - queue_name (str): Name of the queue to subscribe to.
        """
        self.task_id_queue = task_id_queue
        self.mb_server_host_port = mb_server_host_port
        self.vhost = vhost
        self.username = username
        self.passcode = passcode
        self.queue_name = queue_name
        self.conn = None
        self.connect_to_stomp()

    def on_connected(self, frame) -> None:
        """
        Handles connection establishment.

        Parameters:
        - frame: Frame received upon connection.
        """
        logging.info(f"Connected: {frame.headers} {frame.body}")

    def on_error(self, frame) -> None:
        """
        Handles error frames.

        Parameters:
        - frame: Error frame received.
        """
        logging.error(f"Received an error: {frame.body}")

    def on_message(self, frame) -> None:
        """
        Handles incoming messages.

        Parameters:
        - frame: Message frame received.
        """
        try:
            msg = json.loads(frame.body)
            now_ts = time.time()
            if (
                msg["msg_type"] == "task_status"
                and msg["status"] == "defined"
                and now_ts - msg["timestamp"] <= 600
            ):
                task_id = msg["taskid"]
                logging.info(f"Received task ID: {task_id}")
                self.task_id_queue.put(task_id)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")

    def connect_to_stomp(self) -> None:
        """
        Establishes a connection to the STOMP server.

        Raises:
        - Exception: If connection fails after retries.
        """
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                self.conn = stomp.Connection12(
                    [self.mb_server_host_port], vhost=self.vhost
                )
                self.conn.set_listener("", self)
                self.conn.connect(self.username, self.passcode, wait=True)
                logging.info("Connected to STOMP server")

                # Subscribe to the queue
                self.conn.subscribe(destination=self.queue_name, id=1, ack="auto")
                logging.info(f"Subscribed to queue: {self.queue_name}")
                break
            except Exception as e:
                logging.error(f"Error connecting to STOMP server: {e}. Retrying...")
                retries += 1
                time.sleep(1)  # Wait before retrying
