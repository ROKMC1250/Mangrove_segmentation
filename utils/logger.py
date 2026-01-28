import logging
import os
from typing import Optional

def setup_logging(uid: Optional[str] = None):
    """
    Configures logging to output to both console and a file.
    If a UID is provided, logs are saved in a dedicated folder.
    """
    log_dir = "logs"
    if uid:
        log_dir = os.path.join(log_dir, uid)
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, 'train.log')
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s::%(levelname)s::%(name)s::%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def get_logger(name: str, rank: int = 0):
    """
    Returns a logger instance. In a distributed setting, only the main process (rank 0)
    will have an active logger.
    """
    # adapted from https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    class NoOp:
        def __getattr__(self, *args):
            def no_op(*args, **kwargs):
                """Accept every signature by doing non-operation."""
                pass
            return no_op

    if rank == 0:
        return logging.getLogger(name)
    return NoOp()