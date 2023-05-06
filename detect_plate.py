import cv2
from ultralytics import YOLO
import numpy as np

class detect_plate():
    def __init__(self):
        self.model = YOLO('trained_90.pt')

    def get_plate_image(self, image): # KÉP KIEMELŐ AI HASZNÁLATA, MÁS NE LEGYEN ITT
        try:
            result = self.model.predict(source=image, conf=0.5, verbose=False)
            if len(result) == 0:
                print("Error: Could not find licence plate on image")
                return image # Eredeti kép visszaadása

            if len(result) > 1:
                print("Error: More than one licence plate detected on image")
                return image # Eredeti kép visszaadása
            if result is None or result == 0:
                print("Error: Plate returns None")
                return image # Eredeti kép visszaadása
        except:
            print("Error: Plate locating AI not working")
            return image # Eredeti kép visszaadása
        try:
            plate_coordinates = result[0].boxes.xyxy[0].cpu().numpy().astype(int)
            extracted_plate = image[plate_coordinates[1]:plate_coordinates[3], plate_coordinates[0]:plate_coordinates[2]]
        except:
            print("Error: Result converting not working")
            return image # Eredeti kép visszaadása

        if(extracted_plate.shape[0] == 0 or extracted_plate.shape[1] == 0):
            print("Error: Extracted plate size is 0x0")
            return image # Eredeti kép visszaadása

        return self.process_image(extracted_plate)
    
    def process_image(self, image): # KIEMELT KÉP FELDOLGOZÁSA IDE JÖHET

        #kernel2x2 = np.ones((2, 2), np.uint8)
        #im_dilated = cv2.dilate(im_bw, np.ones((2, 2), np.uint8))

        scale_percent = 520/110 # percent of original size
        general_resize_factor = 1.5
        width = int((image.shape[0] * scale_percent)*general_resize_factor)
        height = int((image.shape[0])*general_resize_factor)
        dim = (width, height)
  
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        crop_img = resized[0:resized.shape[0], int(resized.shape[1]*0.16):int(resized.shape[1]*0.96)]

        gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray_image, (5,5), cv2.BORDER_DEFAULT)

        (thresh, im_bw) = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        im_eroded = cv2.erode(im_bw, np.ones((2, 2), np.uint8))

        #blur = cv2.GaussianBlur(crop_img, (3,3), cv2.BORDER_DEFAULT)
        
        return im_eroded

