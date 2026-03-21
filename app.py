from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin

# ✅ FIXED IMPORTS
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


# ✅ HOME
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


# ✅ TRAIN
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


# ✅ PREDICT (MATCHES YOUR HTML BASE64 FLOW)
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']   # base64 string

    # convert base64 → image file
    decodeImage(image, clApp.filename)

    # run prediction
    result = clApp.classifier.predict()

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)


