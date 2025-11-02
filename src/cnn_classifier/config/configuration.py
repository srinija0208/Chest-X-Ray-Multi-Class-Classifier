import os
from cnn_classifier.constants import *
from cnn_classifier.utils.utils import read_yaml, create_directories
from cnn_classifier.entity.config_entity import DataIngestionConfig, BaseModelConfig, ModelTrainingConfig

class ConfigurationManager:

    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)  
        self.params = read_yaml(params_filepath)  

        create_directories([self.config.artifacts_root])  

    ## data ingestion related config
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion    

        create_directories([config.root_dir])  

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,          
            source_url  = config.source_url,      
            local_data_file = config.local_data_file,   
            unzip_dir = config.unzip_dir         
        ) 


        return data_ingestion_config


    ## base model related config

    def get_base_model_config(self)  -> BaseModelConfig:
        config = self.config.base_model

        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_classes = self.params.CLASSES,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_epochs = self.params.EPOCHS,
            params_batch_size = self.params.BATCH_SIZE
        )


        return base_model_config
    

    ## model training related config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        base_model = self.config.base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir)

        create_directories([config.root_dir])

        training_config = ModelTrainingConfig(
            root_dir = Path(config.root_dir),
            trained_model_path = Path(config.trained_model_path),
            updated_base_model_path = Path(base_model.updated_base_model_path),
            training_data = Path(training_data),
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_image_size = params.IMAGE_SIZE,
            params_learning_rate = params.LEARNING_RATE,
            params_is_augmentation = params.AUGMENTATION
        )

        return training_config