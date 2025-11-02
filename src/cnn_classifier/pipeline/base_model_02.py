from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.base_model import BaseModel
from cnn_classifier import logger

    
class BaseModelTrainingPipeline:    
    def __init__(self):
        pass

    def main(self):    

        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModel(config = base_model_config)

        base_model.get_base_model()
        base_model.update_base_model()


if __name__ == "__main__":
    try:

        logger.info(">>>>> stage 02 base model training started <<<<<")

        obj = BaseModelTrainingPipeline()
        obj.main()
        
        logger.info(">>>>> stage 02 base model training completed <<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e

        