"""
Initialize the Data_Pipeline.scripts module.

This module contains the data pipeline scripts for pre-validation,
preprocessing, and post-validation.
"""

# Import from the same directory for local/unit test usage
from .logger import logger
from .utils import (
    setup_gcp_credentials,
    load_bucket_data,
    send_email,
    upload_to_gcs,
)
from .pre_validation import main as pre_validation_main
from .preprocessing import main as preprocessing_main
from .post_validation import post_validation

# Re-export everything for absolute imports
__all__ = [
    "logger",
    "setup_gcp_credentials",
    "load_bucket_data",
    "send_email",
    "upload_to_gcs",
    "pre_validation_main",
    "preprocessing_main",
    "post_validation",
]
