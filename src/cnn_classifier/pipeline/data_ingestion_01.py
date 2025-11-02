from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_ingestion import DataIngestion
from cnn_classifier import logger


class DataIngestionPipeline:
    def __init__(self):
        pass


    def main(self):

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)

        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == "__main__":
    try:

        logger.info(">>>>> stage 01 data ingestion started <<<<<")
        
        obj = DataIngestionPipeline()
        obj.main()

        logger.info(">>>>> stage 01 data ingestion completed <<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e