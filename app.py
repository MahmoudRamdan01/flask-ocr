from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import arabic_reshaper
from bidi.algorithm import get_display
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = Flask(__name__)

# إعدادات النموذج
YOLO_MODEL_PATH = "yolo11m_car_plate_trained.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to('cpu')  # إجبار التشغيل على CPU

# إعدادات PaddleOCR
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='ar',
    use_gpu=False,  # تعطيل GPU
    show_log=False  # إيقاف السجلات الداخلية
)

def crop_license_plate(image_path):
    """قص لوحة الترخيص باستخدام YOLO"""
    try:
        results = yolo_model.predict(source=image_path, conf=0.4)
        if not results:
            return None
            
        image = cv2.imread(image_path)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                continue
                
            # اختيار أكبر صندوق
            max_area = 0
            best_box = None
            for box in boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = box
            
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                cropped = image[y1:y2, x1:x2]
                return cropped
                
        return None
        
    except Exception as e:
        print(f"Error in YOLO processing: {str(e)}")
        return None

def process_ocr(cropped_image):
    """معالجة النص العربي باستخدام PaddleOCR"""
    try:
        result = ocr_engine.ocr(cropped_image)
        texts = []
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                # معالجة النص العربي
                reshaped = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped)
                texts.append(bidi_text)
        return texts
        
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return []

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'لم يتم إرسال صورة'}), 400

        # حفظ الصورة المؤقتة
        image_file = request.files['image']
        temp_dir = "/tmp/ocr_temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, image_file.filename)
        image_file.save(temp_path)

        # معالجة الصورة
        cropped_image = crop_license_plate(temp_path)
        if cropped_image is None:
            return jsonify({'error': 'لم يتم العثور على لوحة ترخيص'}), 404

        # تنفيذ OCR
        detected_text = process_ocr(cropped_image)
        
        # تنظيف الملفات المؤقتة
        os.remove(temp_path)
        
        return jsonify({
            'text': detected_text,
            'message': 'تمت المعالجة بنجاح'
        }), 200

    except Exception as e:
        return jsonify({'error': f'خطأ في الخادم: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
