from flask import Flask, request, jsonify
import os
import cv2
import tempfile
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from ultralytics import YOLO
from paddleocr import PaddleOCR
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Azure Blob Storage Configuration
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=carplatmodels123;AccountKey=f8Fe6VA3N+TcPbdXkRUxH6Pfz+Ynksz0Pu67TSLZeYbCxZcEX7owgkrWgffcaRYajtVSFgXg1UZ7+AStfJfl5Q==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "models"
BLOB_NAME = "yolo11m_car_plate_trained.pt"
MODEL_PATH = "yolo11m_car_plate_trained.pt"

def download_model_from_blob():
    """Download YOLO model from Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)
        
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Azure Blob Storage...")
            with open(MODEL_PATH, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print("Model downloaded successfully")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

# Download model before initializing
download_model_from_blob()
yolo_model = YOLO(MODEL_PATH)

def crop_image_yolo(image_path):
    try:
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
                    return image.crop((x_min, y_min, x_max, y_max))
        return None
    except Exception as e:
        print(f"Error in crop_image_yolo: {str(e)}")
        return None

def detect_text_with_paddleocr(cropped_image):
    try:
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
    except Exception as e:
        print(f"Error in detect_text_with_paddleocr: {str(e)}")
        return []

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        
        # Create temporary directory
        temp_dir = tempfile.gettempdir()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Save uploaded image to temp location
        temp_image_path = os.path.join(temp_dir, image_file.filename)
        image_file.save(temp_image_path)
        
        # Process image
        cropped_image = crop_image_yolo(temp_image_path)
        
        # Cleanup temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        if cropped_image:
            detected_texts = detect_text_with_paddleocr(cropped_image)
            return jsonify({'text': detected_texts}), 200
        else:
            return jsonify({'error': 'License plate not found'}), 404
            
    except Exception as e:
        print(f"Error in ocr_endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
