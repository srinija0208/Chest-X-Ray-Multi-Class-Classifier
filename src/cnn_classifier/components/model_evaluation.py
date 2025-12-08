import tensorflow as tf
from pathlib import Path
import mlflow
import tempfile
import os
import mlflow.keras
from urllib.parse import urlparse

from cnn_classifier.entity.config_entity import ModelEvaluationConfig
from cnn_classifier.utils.utils import save_json


class Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        print(f"DEBUG - Final evaluation score: val_loss={self.score[0]}, val_accuracy={self.score[1]}") 
        self.save_score()

    def save_score(self):
        scores = {"val_loss": self.score[0], "val_accuracy": self.score[1]}
        save_json(path=Path("model_metrics.json"), data=scores)



    def log_into_mlflow(self):
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            
            # 1. Use a temporary directory to save the model
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_model_path = os.path.join(tmpdir, "logged_model.h5")
                
                # 2. Force Keras to save in the compatible HDF5 format
                self.model.save(temp_model_path) 
                
                # 3. Log the temporary model file as a model artifact
                mlflow.log_artifact(temp_model_path, artifact_path="model")
                
                if tracking_url_type_store != "file":
                    # Register the model using the artifact location
                    mlflow.register_model(
                        # The URI points to the run ID and the artifact path "model"
                        f"runs:/{mlflow.active_run().info.run_id}/model/logged_model.h5", 
                        "VGG16Model"
                    )
    