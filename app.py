from flask import Flask, request, jsonify
import os
import cv2
from PIL import Image
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from ultralytics import YOLO
from paddleocr import PaddleOCR
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Azure Storage configuration - مباشرة في الكود
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=carplatmodels123;AccountKey=f8Fe6VA3N+TcPbdXkRUxH6Pfz+Ynksz0Pu67TSLZeYbCxZcEX7owgkrWgffcaRYajtVSFgXg1UZ7+AStfJfl5Q==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "models"
MODEL_BLOB_NAME = "yolo11m_car_plate_trained.pt"

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

def download_model():
    local_model_path = os.path.join(os.getcwd(), MODEL_BLOB_NAME)
    if not os.path.exists(local_model_path):
        print("Downloading model from Azure Storage...")
        blob_client = container_client.get_blob_client(MODEL_BLOB_NAME)
        with open(local_model_path, "wb") as model_file:
            model_file.write(blob_client.download_blob().readall())
        print("Model downloaded successfully")
    return local_model_path

# Initialize YOLO model
YOLO_MODEL_PATH = download_model()
yolo_model = YOLO(YOLO_MODEL_PATH)

def crop_image_yolo(image_path):
    results = yolo_model.predict(source=image_path, conf=0.25)
    image = Image.open(image_path)
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            max_width = -1
            selected_box = None
            for box in result.boxes:
                res = box.xyxy[0]  
                width = res[2].item() - res[0].item()  
                if width > max_width:
                    max_width = width
                    selected_box = res
            if selected_box is not None:
                x_min = selected_box[0].item()
                y_min = selected_box[1].item()
                x_max = selected_box[2].item()
                y_max = selected_box[3].item()
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                return cropped_image
    return None

def detect_text_with_paddleocr(cropped_image):
    ocr = PaddleOCR(use_angle_cls=True, lang='ar')
    image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    results = ocr.ocr(image_cv, cls=True)
    detected_texts = []
    for result in results:
        for (bbox, (text, prob)) in result:
            if text.isdigit():
                reversed_text = text[::-1]
            else:
                reshaped_text = arabic_reshaper.reshape(text)
                reversed_text = get_display(reshaped_text)
            detected_texts.append(reversed_text)
    return detected_texts

@app.route('/')
def home():
    return "OCR Service is running!"

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    image_path = os.path.join(temp_dir, image_file.filename)
    image_file.save(image_path)

    try:
        cropped_image = crop_image_yolo(image_path)
        if cropped_image:
            detected_texts = detect_text_with_paddleocr(cropped_image)
            return jsonify({
                'success': True,
                'text': detected_texts
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'License plate not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
