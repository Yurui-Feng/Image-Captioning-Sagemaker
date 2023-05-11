import json
import requests
import datetime
import boto3
import os
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from sagemaker import Session
from sagemaker.huggingface.model import HuggingFacePredictor
#? add to requirements?
from werkzeug.utils import secure_filename
from flask import send_from_directory

application = Flask(__name__)

application.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Set default uploader path
### Not recommended, ideally file should be uploaded to a S3 bucket
application.config["UPLOAD_FOLDER"] = "uploads"
if not os.path.exists(application.config["UPLOAD_FOLDER"]):
    os.makedirs(application.config["UPLOAD_FOLDER"])

# Read AWS credentials from Elastic Beanstalk environments
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION")

# Read endpoint name from environment
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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.thumbnail(size)
    resized_image.save(output_image_path)

@application.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(application.config["UPLOAD_FOLDER"], filename)

@application.route("/", methods=["GET", "POST"])
def index():
    default_image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
    if request.method == "POST":
        image_url = None
        if "image_url" in request.form:
            image_url = request.form["image_url"]

        # Convert the uploaded file for image captioning
        elif "image_file" in request.files:
            image_file = request.files["image_file"]
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(image_file.filename)
                image_file_path = os.path.join(application.config["UPLOAD_FOLDER"], filename)
                image_file.save(image_file_path)

                # Resize the image
                resized_image_file_path = os.path.join(application.config["UPLOAD_FOLDER"], "resized_" + filename)
                resize_image(image_file_path, resized_image_file_path, (800, 800))  # Resize to 800x800 or whatever size you prefer
                
                image_url = url_for("uploaded_file", filename="resized_" + filename)
                caption = get_image_caption(image_file_path=resized_image_file_path)  # Send the resized image to SageMaker
                return render_template("index.html", caption=caption, image_url=image_url, current_year=datetime.datetime.now().year)

        # URL case
        if image_url:
            caption = get_image_caption(image_url=image_url)
            return render_template("index.html", caption=caption, image_url=image_url, current_year=datetime.datetime.now().year)

    caption = get_image_caption(default_image_url)
    return render_template("index.html", caption=caption, image_url=default_image_url, current_year=datetime.datetime.now().year)

def get_image_caption(image_url=None, image_file_path=None):
    if image_url is not None:
        data = {
            "inputs": [image_url]
        }
    elif image_file_path is not None:
        with open(image_file_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # Encode image bytes to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        data = {
            "inputs": [image_base64]
        }
    else:
        raise ValueError("Both image_url and image_file_path cannot be None.")
    
    response = predictor.predict(data)
    caption = response[0].strip('[]"')
    return caption

if __name__ == "__main__":
    application.run()
# host='0.0.0.0', port=80

