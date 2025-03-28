from ultralytics import YOLO
import cv2
import torch
from PIL import Image
import os 
from util import get_car, read_license_plate
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import numpy as np
import pytesseract as tess
tess.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
custom_config = r'--oem 3 --psm 6'

# load models
license_plate_detector = YOLO('plate_detector.pt').to('cpu') #torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
#license_plate_detector.eval()  # Set the model to evaluation mode

def preprocess_image(img, input_size):
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    return img

""" def get_txt_plate(image):
    input_size = 640  # YOLOv5 input size
    #img = preprocess_image(image, input_size)
    
    with torch.no_grad():
        results = license_plate_detector(image)#[0]
        print(results)
    
    plates = []
    for *box, conf, cls in results:
        if conf > 0.5:  # You can adjust the confidence threshold
            x1, y1, x2, y2 = map(int, box)
            plate = (x1, y1, x2, y2, conf, cls)
            plates.append(plate)
    
    return plates """


def get_txt_plate(frame):
    # detect license plates
    results = license_plate_detector(frame)
    boxes = results[0].boxes.xyxy  # Bounding boxes
    labels = results[0].boxes.cls  # Class labels
    confidences = results[0].boxes.conf  # Confidence scores
    names = results[0].names  # Class names dictionary  
    license_plates = results#.xyxy[0].cpu().numpy() 
    for license_plate in boxes:
        print(license_plate)
        x1, y1, x2, y2 = license_plate
        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
        """ cv2.imshow('Image', license_plate_crop)
        # Wait for a key press indefinitely or for a specified amount of time in milliseconds
        cv2.waitKey(0)
        # Close all OpenCV windows
        cv2.destroyAllWindows() """
        recognized_plate_pil = Image.fromarray(cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB))
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        license_plate_text, _ = read_license_plate(license_plate_crop_thresh)
        text = tess.image_to_string(recognized_plate_pil, config=custom_config, output_type=tess.Output.DICT,lang='eng')
        txt = text.get('text')
        txt = re.sub(r'[^a-zA-Z0-9]', '', txt)
        print(license_plate_text)
        if len(txt)>3:
            return txt

# Test the function
image_path = "sc.png"
plates = get_txt_plate(image_path)
print(plates)
    