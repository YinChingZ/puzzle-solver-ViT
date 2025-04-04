import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def log_training_progress(logger, epoch, train_loss, val_loss):
    """Function to log the training progress."""
    logger.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

def log_evaluation_metrics(logger, metrics):
    """Function to log the evaluation metrics."""
    for metric, value in metrics.items():
        logger.info(f'{metric}: {value}')

def create_log_dir(log_dir):
    """Function to create the log directory if it does not exist."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
