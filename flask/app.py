import json
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.huggingface.model import HuggingFacePredictor


app = Flask(__name__)

# Set up the SageMaker predictor
sagemaker_session = Session()
predictor = HuggingFacePredictor(
    endpoint_name="huggingface-pytorch-inference-2023-04-28-22-15-02-930",
    sagemaker_session=sagemaker_session
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_url = request.form["image_url"]
        caption = get_image_caption(image_url)
        return render_template("index.html", caption=caption)

    return render_template("index.html")

def get_image_caption(image_url):
    data = {
        "inputs": [image_url]
    }
    response = predictor.predict(data)
    caption = response[0]
    return caption


if __name__ == "__main__":
    app.run(debug=True)
