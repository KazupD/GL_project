# Import libraries
import cv2
from ultralytics import YOLO
import numpy as np
import functools

class detect_plate():
    def __init__(self):
        self.model = YOLO('trained_90.pt')

    def get_plate_image(self, image): # KÉP KIEMELŐ AI HASZNÁLATA, MÁS NE LEGYEN ITT
        try:
            result = self.model.predict(source=image, conf=0.5, verbose=False)
            if len(result) == 0:
                print("Error: Could not find licence plate on image")
                return None

            if len(result) > 1:
                print("Error: More than one licence plate detected on image")
                return None
            if result is None or result == 0:
                print("Error: Plate returns None")
                return None
        except:
            print("Error: Plate locating AI not working")
            return None
        try:
            plate_coordinates = result[0].boxes.xyxy[0].cpu().numpy().astype(int)
            extracted_plate = image[plate_coordinates[1]:plate_coordinates[3], plate_coordinates[0]:plate_coordinates[2]]
        except:
            print("Error: Result converting not working")
            return None

        if(extracted_plate.shape[0] == 0 or extracted_plate.shape[1] == 0):
            print("Error: Extracted plate size is 0x0")
            return None

        return self.process_image(extracted_plate) # image
    
    def process_image(self, image): # KIEMELT KÉP FELDOLGOZÁSA IDE JÖHET

        resized = self.perscpective_correction(image=image, general_resize_factor=1)

        crop_img = resized[int(resized.shape[0]*0.08):int(resized.shape[0]*0.92), int(resized.shape[1]*0.16):int(resized.shape[1]*0.96)]

        unisize = cv2.resize(crop_img, (440, 110), interpolation = cv2.INTER_AREA)

        gray_image = cv2.cvtColor(unisize, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray_image, (9,9), cv2.BORDER_DEFAULT)

        #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)

        
        return blur # image
    
    def perscpective_correction(self, image, general_resize_factor):
        scale_percent = 520/110 # percent of original size
        initial_width = image.shape[1]
        initial_height = image.shape[0]

        width = int((initial_height * scale_percent)*general_resize_factor)
        height = int(initial_height*general_resize_factor)
        dim = (width, height)
  
        horizontally_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        return horizontally_resized # image
    
    def pad_image(src, padding):
        top = int(padding * src.shape[0])  # shape[0] = rows
        bottom = top
        left = int(padding * src.shape[1])  # shape[1] = cols
        right = left
        
        dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_REPLICATE)

        return dst


