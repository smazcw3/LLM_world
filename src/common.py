import os
import logging
from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName('Databricks Shell')
         .getOrCreate()
        )

logger = logging.getLogger()
logger.setLevel(logging.INFO)