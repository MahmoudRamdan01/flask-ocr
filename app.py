from flask import Flask, request, jsonify
import os
import cv2
from PIL import Image
import numpy as np
import tempfile
import arabic_reshaper
from bidi.algorithm import get_display
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = Flask(__name__)

YOLO_MODEL_PATH = os.path.join(os.getcwd(), "yolo11m_car_plate_trained.pt")
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

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    temp_dir = tempfile.gettempdir()  
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, image_file.filename)
    image_file.save(image_path)

    cropped_image = crop_image_yolo(image_path)
    if cropped_image:
        detected_texts = detect_text_with_paddleocr(cropped_image)
        return jsonify({'text': detected_texts}), 200
    else:
        return jsonify({'error': 'License plate not found'}), 404

if __name__ == '__main__':
    app.run()
