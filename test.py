from ultralytics import YOLO
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import os

# --- Configuration ---
# Update these paths to point to your model and image filbne
MODEL_PATH = r"C:\Coding\Python\DS_AI_ML\qrfind\model\v1_bnq.pt"
IMAGE_PATH = r"C:\Coding\Python\DS_AI_ML\qrfind\archive\v2VCard-Title_png.rf.23f56d6f4c253e50d5e244fef6db556e.jpg"
OUTPUT_IMAGE_PATH = "result.jpg"
# --- End of Configuration ---

def decode_qr_from_image(model_path, image_path):
    """
    Detects and decodes QR codes in an image using a YOLO model.
    Uses an advanced multi-stage decoding pipeline for better results on damaged codes.
    Saves the output with annotations to a file.

    Args:
        model_path (str): The path to the trained YOLO model file.
        image_path (str): The path to the input image.
    """
    # 1. Load the pre-trained YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image file at '{image_path}'")
        return

    # 3. Run model inference to get bounding boxes
    try:
        results = model(img, verbose=False)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return

    print(f"Found {len(results[0].boxes)} potential objects.")

    # 4. Iterate over each detected bounding box
    for result in results:
        boxes = result.boxes.cpu().numpy()
        if not len(boxes):
            print("No bounding boxes were detected in the image.")
            continue

        for box in boxes:
            coords = box.xyxy[0].astype(int)
            x1, y1, x2, y2 = coords
            padding = 15
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(img.shape[1], x2 + padding), min(img.shape[0], y2 + padding)
            
            cropped_qr_code = img[y1:y2, x1:x2]
            gray_qr = cv2.cvtColor(cropped_qr_code, cv2.COLOR_BGR2GRAY)
            
            qr_data = None

            # --- Decoding Pipeline ---

            # Attempt 1: Pyzbar with simple thresholding
            _, thresh_qr = cv2.threshold(gray_qr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if decode(thresh_qr):
                qr_data = decode(thresh_qr)[0].data.decode('utf-8')

            # Attempt 2: OpenCV's QRCodeDetector
            if not qr_data:
                print("Attempt 1 (Pyzbar) failed. Trying Attempt 2 (OpenCV Detector)...")
                detector = cv2.QRCodeDetector()
                data, _, _ = detector.detectAndDecode(gray_qr)
                if data:
                    qr_data = data

            # Attempt 3: Pyzbar with morphological closing
            if not qr_data:
                print("Attempt 2 (OpenCV) failed. Trying Attempt 3 (Morphological Closing)...")
                kernel = np.ones((2, 2), np.uint8)
                closed_qr = cv2.morphologyEx(thresh_qr, cv2.MORPH_CLOSE, kernel, iterations=1)
                if decode(closed_qr):
                    qr_data = decode(closed_qr)[0].data.decode('utf-8')
            
            # Attempt 4: Pyzbar with Upscaling and Adaptive Thresholding
            if not qr_data:
                print("Attempt 3 failed. Trying Attempt 4 (Upscaling & Adaptive Threshold)...")
                # Upscale the image
                upscaled_qr = cv2.resize(gray_qr, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                # Apply Gaussian blur
                blurred_qr = cv2.GaussianBlur(upscaled_qr, (5, 5), 0)
                # Apply adaptive thresholding
                adaptive_thresh_qr = cv2.adaptiveThreshold(blurred_qr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                if decode(adaptive_thresh_qr):
                    qr_data = decode(adaptive_thresh_qr)[0].data.decode('utf-8')

            # --- Drawing Results ---
            draw_x1, draw_y1, draw_x2, draw_y2 = box.xyxy[0].astype(int)
            if qr_data:
                print(f"✅ Decoded Data: {qr_data}")
                cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                cv2.putText(img, qr_data, (draw_x1, draw_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print("❌ All decoding attempts failed for a bounding box.")
                cv2.rectangle(img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 0, 255), 2)
                cv2.putText(img, "DECODE FAILED", (draw_x1, draw_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 8. Save the final image with annotations
    try:
        cv2.imwrite(OUTPUT_IMAGE_PATH, img)
        print(f"\nSuccessfully saved the output image to: {os.path.abspath(OUTPUT_IMAGE_PATH)}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    decode_qr_from_image(MODEL_PATH, IMAGE_PATH)

