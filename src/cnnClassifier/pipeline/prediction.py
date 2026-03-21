import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
        # ✅ Load model ONLY ONCE
        self.model = load_model(
            os.path.join("model", "model.h5"),
            compile=False
        )

    def predict(self):
        # load image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)

        # ✅ IMPORTANT (same as training)
        test_image = test_image / 255.0

        # expand dims
        test_image = np.expand_dims(test_image, axis=0)

        # prediction
        result = np.argmax(self.model.predict(test_image), axis=1)

        print("Prediction index:", result)

        # ⚠️ Try BOTH mappings once (see which is correct)
        if result[0] == 1:
            prediction = 'Tumor'
        else:
            prediction = 'Normal'

        return [{"image": prediction}]