from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from shreyanshj01_utils.utils import decodeImage
from detect import Predictor

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.obj_detect = Predictor()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.obj_detect.run_inference()
    return jsonify(result)


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 8000
    app.run(host='127.0.0.1', port=port)
    #app.run(host='0.0.0.0', port=7000, debug=True)