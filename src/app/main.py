"""Main entry point for the Stock-Sent-Earnings module.

This script initializes the service, sets up logging, and starts
consuming messages from the configured message queue for earnings sentiment analysis.
"""

import os
import sys

# Add 'src/' to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.queue_handler import consume_messages
from app.utils.setup_logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def main() -> None:
    """Starts the Earnings Sentiment Analysis Service by consuming messages
    and analyzing sentiment around earnings-related events.
    
    This service listens to messages from a queue (RabbitMQ or SQS),
    applies sentiment analysis, and publishes the results to an output system.


    
    """
    logger.info("Starting Earnings Sentiment Analysis Service...")
    consume_messages()


if __name__ == "__main__":
    main()
