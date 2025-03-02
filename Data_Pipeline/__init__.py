# Import modules to make them available when importing from Data_Pipeline
# Try to use relative imports first, then fall back to absolute if needed
try:
    from .scripts import (
        logger,
        post_validation,
        pre_validation,
        preprocessing,
        utils,
    )
except ImportError:
    # This branch handles cases where the package is imported from outside
    # its own directory structure
    from Data_Pipeline.scripts import (
        logger,
        post_validation,
        pre_validation,
        preprocessing,
        utils,
    )
