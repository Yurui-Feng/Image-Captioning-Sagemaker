import json
import requests
import datetime
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
    endpoint_name="huggingface-pytorch-inference-2023-05-01-00-45-11-156",
    sagemaker_session=sagemaker_session
)

@app.route("/", methods=["GET", "POST"])
def index():
    default_image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
    if request.method == "POST":
        image_url = request.form["image_url"]
        caption = get_image_caption(image_url)
        return render_template("index.html", caption=caption, image_url=image_url, current_year=datetime.datetime.now().year)

    caption = get_image_caption(default_image_url)
    return render_template("index.html", caption=caption, image_url=default_image_url, current_year=datetime.datetime.now().year)


def get_image_caption(image_url):
    data = {
        "inputs": [image_url]
    }
    response = predictor.predict(data)
    caption = response[0].strip('[]"')
    return caption


if __name__ == "__main__":
    app.run(debug=True)
