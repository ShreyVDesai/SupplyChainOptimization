import logging
from airflow.utils.log.logging_mixin import LoggingMixin


# Create a logger that inherits from Airflow's LoggingMixin
class AirflowLogger:
    """
    Logger class that uses Airflow's logging system.
    This replaces the previous SCO logging implementation.
    """

    def __init__(self, name="SupplyChainOptimization"):
        # Create a logging mixin instance
        self._logging_mixin = LoggingMixin()
        # Store the logger name
        self._logger_name = name

    def info(self, msg, *args, **kwargs):
        self._logging_mixin.log.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logging_mixin.log.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logging_mixin.log.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logging_mixin.log.debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logging_mixin.log.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logging_mixin.log.exception(msg, *args, **kwargs)


# Create a single instance of the logger to be imported by other modules
logger = AirflowLogger()
