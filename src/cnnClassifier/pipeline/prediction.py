import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
        # ✅ Load model once
        self.model = load_model(
            os.path.join("model", "model.h5"),
            compile=False
        )

    def predict(self):
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)

        # Normalize
        test_image = test_image / 255.0

        # Expand dims
        test_image = np.expand_dims(test_image, axis=0)

        # Prediction
        pred = self.model.predict(test_image)

        confidence = float(pred[0][0])

        print("Confidence:", confidence)

        if confidence > 0.5:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return {
            "prediction": prediction,
            "confidence": confidence
        }