import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # ✅ CRITICAL FIX: Recompile model with fresh optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )


    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # ✅ FIX: correct paths
        train_path = os.path.join(self.config.training_data, "train")
        val_path = os.path.join(self.config.training_data, "val")

        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.2,
            horizontal_flip=True,
            **datagenerator_kwargs
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # ✅ TRAIN GENERATOR
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=train_path,
            shuffle=True,
            **dataflow_kwargs
        )

        # ✅ VALIDATION GENERATOR
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=val_path,
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )