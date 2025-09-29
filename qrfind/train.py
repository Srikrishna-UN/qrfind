from ultralytics import YOLO


model = YOLO(r'C:\Coding\Python\DS_AI_ML\qrfind\model\yolo11s.pt')

if __name__ == '__main__':
    results = model.train(data='config.yaml', epochs=100, imgsz=640)