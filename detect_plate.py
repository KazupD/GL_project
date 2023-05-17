import cv2
from ultralytics import YOLO
import numpy as np
import functools

class detect_plate():
    def __init__(self):
        self.model_extract_plate = YOLO('trained_90.pt')
        self.model_extract_char = YOLO('yolo_custom_char_250.pt')

    def get_plate_image(self, image): # KÉP KIEMELŐ AI HASZNÁLATA, MÁS NE LEGYEN ITT
        try:
            result = self.model_extract_plate.predict(source=image, conf=0.5, verbose=False)
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

        crop_img = resized[int(resized.shape[0]*0.02):int(resized.shape[0]*0.98), int(resized.shape[1]*0.16):int(resized.shape[1]*0.98)] 

        unisize = cv2.resize(crop_img, (440, 110), interpolation = cv2.INTER_AREA)

        gray_image = cv2.cvtColor(unisize, cv2.COLOR_BGR2GRAY)

        #blur = cv2.GaussianBlur(gray_image, (5, 5), cv2.BORDER_DEFAULT)

        #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 15)
        ret3, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        corners = self.get_corners_of_text(image=unisize)
        if(corners is not None):
            transformed_image = self.do_perspective_transform(image=thresh, text_box_coordinates=corners)
            extended = self.pad_image(transformed_image, 0.1, color="white")
            dilated = cv2.dilate(extended, np.ones((3, 3), np.uint8))
            opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            #cv2.imshow('image', opening)
            #cv2.waitKey()
            return opening # image
        else:
            return gray_image
    

    def perscpective_correction(self, image, general_resize_factor):
        scale_percent = 520/110 # percent of original size
        initial_width = image.shape[1]
        initial_height = image.shape[0]

        width = int((initial_height * scale_percent)*general_resize_factor)
        height = int(initial_height*general_resize_factor)
        dim = (width, height)
  
        horizontally_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        return horizontally_resized # image
    

    def get_corners_of_text(self, image): # Karakter kép kiemelése a rendszám képéről
        try: 
            results = self.model_extract_char.predict(source=image, conf=0.5, verbose=False)
        except Exception as e:
            print("Error: Character locating AI not working")
            print(e)
            return None
        try:
            sorted_char_coordinates = sorted(results[0].boxes.xyxy, key=lambda x: x.cpu().numpy().astype(int)[0])
            first_char_coordinates = sorted_char_coordinates[0].cpu().numpy().astype(int)
            last_char_coordinates = sorted_char_coordinates[-1].cpu().numpy().astype(int)
        except Exception as e:
            print("Error: Could not give text bounding box coordinates")
            print(e)
            return None

        return (first_char_coordinates, last_char_coordinates)
    
    
    def do_perspective_transform(self, image, text_box_coordinates):
        if(text_box_coordinates is not None):
            src = np.float32([[text_box_coordinates[0][0],text_box_coordinates[0][1]], [text_box_coordinates[1][2],text_box_coordinates[1][1]],
                                [text_box_coordinates[0][0],text_box_coordinates[0][3]], [text_box_coordinates[1][2],text_box_coordinates[1][3]]])
            dst = np.float32([[0, 0], [440, 0],
                                [0, 110], [440, 110]])
            matrix = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(image, matrix, (440, 110), borderMode=cv2.BORDER_CONSTANT, borderValue = [230, 230, 230])
            return result
        else: return image


    def pad_image(self, src, padding, color=None):
        top = int(padding * src.shape[0])  # shape[0] = rows
        bottom = top
        left = int(padding * src.shape[1])  # shape[1] = cols
        right = left
        if(color == None):
            dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_REPLICATE)
        elif(color == "white"):
            dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        elif(color == "black"):
            dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        return dst


