from cnn_classifier.pipeline.data_ingestion_01 import DataIngestionPipeline
from cnn_classifier.pipeline.base_model_02 import BaseModelTrainingPipeline
from cnn_classifier.pipeline.model_training_03 import ModelTrainingPipeline

from cnn_classifier import logger


stage_name = "data ingestion"

try:

    logger.info(">>>>> stage 01 {stage_name} started <<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(">>>>> stage 01 {stage_name} completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e



stage_name = "base model training"

try:
    
    logger.info(f">>>>> stage 02 {stage_name} started <<<<<")
    obj = BaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage 02 {stage_name} completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e


stage_name = "model training"

try:

    logger.info(f">>>>> stage 03 {stage_name} started <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage 03 {stage_name} completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e