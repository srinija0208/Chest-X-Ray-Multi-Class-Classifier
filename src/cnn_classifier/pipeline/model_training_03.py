from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_training import Training
from cnn_classifier import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer = Training(config=model_training_config)

        model_trainer.get_base_model()
        model_trainer.train_valid_generator()
        model_trainer.train()


if __name__ == "__main__":
    try:

        logger.info(">>>>>> Model Training Pipeline started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(">>>>>> Model Training Pipeline completed <<<<<<\n\nx==========x")
        
    except Exception as e:
        logger.exception(e)
        raise e