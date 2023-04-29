import os
import json
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


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
    if content_type == "application/json":
        input_data = json.loads(input_data)
        return input_data
    else:
        raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(input_data, model_artifacts):
    model = model_artifacts["model"]
    feature_extractor = model_artifacts["feature_extractor"]
    tokenizer = model_artifacts["tokenizer"]
    device = model_artifacts["device"]

    image_urls = input_data["inputs"]
    images = []

    for image_url in image_urls:
        response = requests.get(image_url)
        i_image = Image.open(BytesIO(response.content))
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
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
