import os
import json
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import base64
import binascii
import re

def model_fn(model_dir):
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    feature_extractor = ViTImageProcessor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return {
        "model": model,
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "device": device,
    }

def input_fn(input_data, content_type):
    def is_url(input_data):
        return input_data.startswith("http://") or input_data.startswith("https://")

    if content_type == "application/json":
        input_data = json.loads(input_data)["inputs"][0]

        if is_url(input_data):
            # Load image from URL
            response = requests.get(input_data)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from base64 string
            image_bytes = base64.b64decode(input_data)
            image = Image.open(BytesIO(image_bytes))
        return image
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))


def predict_fn(input_image, model_artifacts):
    model = model_artifacts["model"]
    feature_extractor = model_artifacts["feature_extractor"]
    tokenizer = model_artifacts["tokenizer"]
    device = model_artifacts["device"]

    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    pixel_values = feature_extractor(images=input_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds



def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept: {accept}")
