import logging
import os

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.init(*args, **kwargs)
        return cls._instance

    def init(
        self,
        logger_name,
        log_dir_path,
        log_file_name="app.log",
        log_level=logging.DEBUG,
        console_level=logging.INFO,
    ):
        self.logger = self.configure_logger(
            logger_name, log_dir_path, log_file_name, log_level, console_level
        )

    @staticmethod
    def configure_logger(
        logger_name,
        log_dir_path,
        log_file_name="app.log",
        log_level=logging.DEBUG,
        console_level=logging.INFO,
    ):
        """
        Configures a logger with specified name and log file path.

        Parameters:
        - logger_name: Name of the logger.
        - log_dir_path: Directory path for the log file.
        - log_file_name: Name of the log file (default: 'app.log').
        - log_level: Logging level for the file handler (default: DEBUG).
        - console_level: Logging level for the console handler (default: INFO).

        Returns:
        - A configured logger object.
        """
        # Ensure the log directory exists
        os.makedirs(log_dir_path, exist_ok=True)

        # Construct the full log file path
        log_file_path = os.path.join(log_dir_path, log_file_name)

        # Create a logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = False  # Disable propagation to avoid duplicate logs

        # Remove existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        # Create a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(console_level)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Log a message to ensure the logger is working
        logger.info(f"Logger initialized. Logging to file: {log_file_path}")

        return logger

    def get_logger(self):
        return self.logger

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# Example usage
#logger = Logger('demo_logger', '/data/model-data/logs', 'pred.log')
#logger.get_logger().debug("This is a debug message.")
#logger.get_logger().info("This is an info message.")
#logger.get_logger().warning("This is a warning message.")
#logger.get_logger().error("This is an error message.")
#logger.get_logger().critical("This is a critical message.")

