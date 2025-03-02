import logging
import sys

# Create a simple formatter for logs
formatter = logging.Formatter(
    "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
)


class Logger:
    """
    Simple logger that uses Airflow's logging system when available,
    or falls back to standard Python logging otherwise.
    """

    def __init__(self, name="SupplyChainOptimization"):
        # Create a standard Python logger as fallback
        self._standard_logger = logging.getLogger(name)
        self._standard_logger.setLevel(logging.INFO)

        # Add stdout handler if not already present
        if not self._standard_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            self._standard_logger.addHandler(handler)

        # Try to import Airflow's LoggingMixin if available
        try:
            from airflow.utils.log.logging_mixin import LoggingMixin

            self._airflow_logger = LoggingMixin().log
            self._use_airflow = True
        except ImportError:
            self._use_airflow = False

    def _get_logger(self):
        return (
            self._airflow_logger
            if self._use_airflow
            else self._standard_logger
        )

    def info(self, msg, *args, **kwargs):
        self._get_logger().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._get_logger().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._get_logger().error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._get_logger().debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._get_logger().critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._get_logger().exception(msg, *args, **kwargs)

    def setLevel(self, level):
        """
        Set the logging level of the logger.

        Args:
            level: The logging level to set (can be a string like 'DEBUG' or a logging constant)
        """
        if isinstance(level, str):
            # Convert string level to logging constant
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            level = level_map.get(level, logging.INFO)

        # Set level on standard logger
        self._standard_logger.setLevel(level)


# Create a single instance of the logger to be imported by other modules
logger = Logger()
