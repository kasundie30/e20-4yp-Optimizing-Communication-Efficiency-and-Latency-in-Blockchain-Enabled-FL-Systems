import logging
import json
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Injects timestamp, level, bank_id, round_num, and component.
    """
    def __init__(self, bank_id: str = "Unknown", component: str = ""):
        super().__init__()
        self.bank_id = bank_id
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        # Extract default fields
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "bank_id": getattr(record, "bank_id", self.bank_id),
            "round_num": getattr(record, "round_num", None),
            "component": getattr(record, "component", self.component or record.name),
            "message": record.getMessage()
        }
        
        # Include exception traceback if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)


def setup_logging(bank_id: str, component: str = "", level: int = logging.INFO) -> logging.Logger:
    """
    Configures the root logger to output structured JSON to stdout.
    
    Args:
        bank_id:   Identifier for the node (e.g., "BankA")
        component: Logical component name
        level:     Logging level (default logging.INFO)
    
    Returns:
        The configured root logger.
    """
    root_logger = logging.getLogger()
    
    # Remove all existing handlers to ensure we don't duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = JSONFormatter(bank_id=bank_id, component=component)
    handler.setFormatter(formatter)
    
    root_logger.addHandler(handler)
    
    return root_logger
