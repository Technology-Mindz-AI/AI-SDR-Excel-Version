import logging

# Logging setup
LOG_FILE = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # optional: still outputs to console
    ]
)

# Create a logger that can be imported by other modules
logger = logging.getLogger(__name__)