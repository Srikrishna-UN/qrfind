# QR Code Detection and Decoding Engine

This project provides a complete pipeline for detecting and decoding QR codes from images, including those that are damaged or distorted. It uses a custom-trained **YOLOv8 model** for high-accuracy detection and a robust, multi-stage decoding process to extract the QR code's value. The final output is provided in structured JSON files, ready for downstream use.

---

## Features

* **High-Accuracy Detection**: Utilizes a custom-trained YOLOv8 model to locate QR codes within images.
* **Robust Decoding**: Employs a multi-stage pipeline using `pyzbar` and OpenCV with advanced image processing techniques (thresholding, morphological operations, upscaling) to decode even damaged or low-quality QR codes.
* **Batch Processing**: Capable of processing an entire folder of images in a single run.
* **Structured JSON Output**: Generates two distinct JSON files: one with bounding boxes only and another with both bounding boxes and their decoded values.
* **Complete Workflow**: Includes helper scripts for converting annotations and splitting datasets for training.

---

## Project Workflow

The project is divided into two main phases: **Training** (only required once) and **Inference**.

1. **Data Preparation**: Raw image annotations in JSON format are converted into the YOLO `.txt` format.
2. **Dataset Splitting**: The prepared dataset is automatically split into training and validation sets.
3. **Model Training**: A YOLOv11 model is trained on the prepared dataset to learn how to detect QR codes.
4. **Inference & Decoding**: The trained model is used to perform inference on a folder of new images. For each detected QR code, the multi-stage decoding pipeline is executed to extract its value.
5. **Output Generation**: The results are saved into two final JSON files.

---

## Directory Structure

Your folders should be organized as follows:

```
qr_code_project/
│
├── model/
│   └── v1_bnq.pt      
│
├── config.yaml        
├── test.py
├── train.py
├── preprocess.py    
├── main.py        
└── README.md
```

---

## Setup and Installation

1. Clone the repository and navigate to the project directory.

2. Create a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running Inference

The script will process all images in the folder and generate two output files in your project's root directory:

* `submission_detection_1.json`: Contains only the `image_id` and bounding box coordinates for each detected QR code.
* `submission_detection_2.json`: Contains the `image_id`, bounding box, and the decoded value for each QR code.

---

## Example JSON Output

**submission_detection_1.json**

```json
[
  {
    "image_id": "img060",
    "qrs": [
      {
        "bbox": [123, 456, 789, 1011]
      }
    ]
  }
]
```

**submission_detection_2.json**

```json
[
  {
    "image_id": "img060",
    "qrs": [
      {
        "bbox": [123, 456, 789, 1011],
        "value": "XYZ"
      }
    ]
  }
]
```
