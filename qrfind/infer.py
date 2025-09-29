from ultralytics import YOLO
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import os
import argparse
import json

MODEL_PATH = r"C:\Coding\Python\DS_AI_ML\qrfind\model\v1_bnq.pt"

def process_image_for_qrs(model_path, image_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    img = cv2.imread(image_path)
    image_id = os.path.basename(image_path).split('.')[0]
    if img is None:
        return {
            "image_id": image_id,
            "qrs": [],
            "error": f"Could not read the image file at '{image_path}'"
        }

    try:
        results = model(img, verbose=False)
    except Exception as e:
        return {"image_id": image_id, "qrs": [], "error": f"Model inference failed: {e}"}

    output_data = {"image_id": image_id, "qrs": []}
    boxes = results[0].boxes.cpu().numpy()

    for box in boxes:
        coords = box.xyxy[0].astype(int)
        bbox_list = coords.tolist()
        x1, y1, x2, y2 = coords

        padding = 15
        crop_x1, crop_y1 = max(0, x1 - padding), max(0, y1 - padding)
        crop_x2, crop_y2 = min(img.shape[1], x2 + padding), min(img.shape[0], y2 + padding)
        
        cropped_qr_code = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped_qr_code.shape[0] == 0 or cropped_qr_code.shape[1] == 0:
            continue
        
        gray_qr = cv2.cvtColor(cropped_qr_code, cv2.COLOR_BGR2GRAY)
        qr_data = None

        _, thresh_qr = cv2.threshold(gray_qr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if decode(thresh_qr):
            qr_data = decode(thresh_qr)[0].data.decode('utf-8')

        if not qr_data:
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(gray_qr)
            if data:
                qr_data = data

        if not qr_data:
            kernel = np.ones((2, 2), np.uint8)
            closed_qr = cv2.morphologyEx(thresh_qr, cv2.MORPH_CLOSE, kernel, iterations=1)
            if decode(closed_qr):
                qr_data = decode(closed_qr)[0].data.decode('utf-8')

        if not qr_data:
            upscaled = cv2.resize(gray_qr, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
            adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            if decode(adaptive_thresh):
                qr_data = decode(adaptive_thresh)[0].data.decode('utf-8')

        output_data["qrs"].append({
            "bbox": bbox_list,
            "value": qr_data
        })

    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and decode QR codes in an image, then output the results as a JSON string."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="The full path to the input image file."
    )
    args = parser.parse_args()

    final_result = process_image_for_qrs(MODEL_PATH, args.image_path)

    print(json.dumps(final_result, indent=2))

