import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import io
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch

# Initialize FastAPI app
app = FastAPI()

# Load YOLOS-Fashionpedia model
model_name = "valentinafeve/yolos-fashionpedia"
model = YolosForObjectDetection.from_pretrained(model_name)
processor = YolosImageProcessor.from_pretrained(model_name)


# ---------- COLOR DETECTION FUNCTION ----------
def get_dominant_color(image, k=4):
    try:
        image = image.convert("RGB")  # Ensure 3-channel RGB
        image = image.resize((64, 64))  # Resize to simplify
        image_np = np.array(image)

        reshaped = image_np.reshape(-1, 3)  # Flatten to (N, 3)

        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(reshaped)
        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

        return tuple(int(c) for c in dominant_color)
    
    except Exception as e:
        print("Error in get_dominant_color():", e)
        raise


# ---------- HOME ROUTE ----------
@app.get("/")
def home():
    return {"message": "Clothing detection API is running with YOLOS-Fashionpedia!"}


# ---------- CLOTHING + COLOR DETECTION ROUTE ----------
@app.post("/detect")
async def detect_clothing(file: UploadFile = File(...)):
    # Load image from upload
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess for YOLOS
    inputs = processor(images=pil_image, return_tensors="pt")

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits (class predictions) and bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Convert logits → labels
    probabilities = logits.softmax(-1)[0, :, :-1]  # drop background
    scores, labels = probabilities.max(-1)

    detected_items = []
    for score, label, box in zip(scores, labels, bboxes[0]):
        if score > 0.5:  # Confidence threshold
            detected_items.append(model.config.id2label[label.item()])

    # Remove duplicates
    detected_items = list(set(detected_items))

    # Get dominant color
    rgb = get_dominant_color(pil_image)

    # Convert RGB → HEX
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

    return {
        "detected_items": detected_items,
        "dominant_color": hex_color
    }
