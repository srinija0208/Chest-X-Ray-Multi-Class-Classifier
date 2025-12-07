from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_evaluation import Evaluation
from cnn_classifier import logger


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        evaluation = Evaluation(model_evaluation_config)
        evaluation.evaluation()
        # evaluation.log_into_mlflow()


if __name__ == "__main__":

    try:

        logger.info(">>>>>> Model Evaluation Pipeline started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(">>>>>> Model Evaluation Pipeline completed <<<<<<\n\nx==========x")
        
    except Exception as e:
        
        logger.exception(e)
        raise e