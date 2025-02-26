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
        logger_name: str,
        log_dir_path: str,
        log_file_name: str = "app.log",
        log_level: int = logging.DEBUG,
        console_level: int = logging.INFO,
    ):
        """
        Initializes the Logger instance.

        Parameters:
        - logger_name (str): Name of the logger.
        - log_dir_path (str): Directory path for the log file.
        - log_file_name (str): Name of the log file. Defaults to 'app.log'.
        - log_level (int): Logging level for the file handler. Defaults to DEBUG.
        - console_level (int): Logging level for the console handler. Defaults to INFO.
        """
        self.logger = self.configure_logger(logger_name, log_dir_path, log_file_name, log_level, console_level)

    @staticmethod
    def configure_logger(
        logger_name: str,
        log_dir_path: str,
        log_file_name: str = "app.log",
        log_level: int = logging.DEBUG,
        console_level: int = logging.INFO,
    ) -> logging.Logger:
        """
        Configures a logger with specified name and log file path.

        Parameters:
        - logger_name (str): Name of the logger.
        - log_dir_path (str): Directory path for the log file.
        - log_file_name (str): Name of the log file. Defaults to 'app.log'.
        - log_level (int): Logging level for the file handler. Defaults to DEBUG.
        - console_level (int): Logging level for the console handler. Defaults to INFO.

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
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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

    def get_logger(self) -> logging.Logger:
        """Returns the configured logger."""
        return self.logger

    def debug(self, message: str):
        """Logs a debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Logs an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Logs an error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Logs a critical message."""
        self.logger.critical(message)


# if __name__ == "__main__":
#     logger = Logger()
#     logger.init("my_logger", "/path/to/log/directory")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
#     logger.critical("This is a critical message.")
