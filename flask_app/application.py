import json
import requests
import datetime
import boto3
import os
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from sagemaker import Session
from sagemaker.huggingface.model import HuggingFacePredictor

application = Flask(__name__)

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION")
endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")

# Create a boto3 session with the provided credentials and region
boto3_session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Set up the SageMaker predictor
sagemaker_session = Session(boto3_session)
predictor = HuggingFacePredictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session
)

@application.route("/", methods=["GET", "POST"])
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
    application.run(host='0.0.0.0', port=80)
