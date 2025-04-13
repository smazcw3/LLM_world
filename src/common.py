import os
import logging
from pyspark.sql import SparkSession
import shutil
    

def create_spark_session(app_name="Databricks Shell"):
    """
    Creates and returns a Spark session with the specified app name.
    
    Args:
        app_name (str): Name for the Spark application
        
    Returns:
        SparkSession: Configured Spark session
    """
    spark = (SparkSession.builder
             .appName(app_name)
             .getOrCreate()
            )
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return spark


def cleanup(data_dir):
    """
    Removes all files and directories within data_dir except for __init__.py.
    
    Args:
        data_dir (str): Path to directory to clean
    """
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if item == "__init__.py":
            continue
        # Remove everything else
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)