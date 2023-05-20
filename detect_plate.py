# Import libraries
import cv2
from ultralytics import YOLO
import numpy as np

class detect_plate():
    def __init__(self):
        self.model_extract_plate = YOLO('trained_90.pt')
        self.model_extract_char = YOLO('yolo_custom_char_250.pt')

    # This function search and crop the license plate 
    def get_plate_image(self, image):
        try:
            result = self.model_extract_plate.predict(source=image, conf=0.5, verbose=False)
            # There is no number plate in the picture
            if len(result) == 0:
                print("Error: Could not find licence plate on image")
                return None
            # There is more than one number plate in the picture
            if len(result) > 1:
                print("Error: More than one licence plate detected on image")
                return None
            # YOLO error
            if result is None or result == 0: 
                print("Error: Plate returns None")
                return None
        except:
            # There is a problem with YOLO
            print("Error: Plate locating AI not working")
            return None

        try:
            # Crop the image
            plate_coordinates = result[0].boxes.xyxy[0].cpu().numpy().astype(int)
            extracted_plate = image[plate_coordinates[1]:plate_coordinates[3], plate_coordinates[0]:plate_coordinates[2]]
        except:
            print("Error: Result converting not working")
            return None

        if(extracted_plate.shape[0] == 0 or extracted_plate.shape[1] == 0):
            print("Error: Extracted plate size is 0x0")
            return None

        return self.process_image(extracted_plate) # image

    # This processing makes the cropped image more suitable for character recognition.
    def process_image(self, image): # KIEMELT KÉP FELDOLGOZÁSA IDE JÖHET
        # Perspective projection size correction
        resized = self.perscpective_correction(image=image, general_resize_factor=1)
        # Trims the unwanted edges
        crop_img = resized[int(resized.shape[0]*0.02):int(resized.shape[0]*0.98), int(resized.shape[1]*0.16):int(resized.shape[1]*0.98)] 
        # Aspect ration correction
        unisize = cv2.resize(crop_img, (440, 110), interpolation = cv2.INTER_AREA)
        # Grayscale conversion
        gray_image = cv2.cvtColor(unisize, cv2.COLOR_BGR2GRAY)

        #blur = cv2.GaussianBlur(gray_image, (5, 5), cv2.BORDER_DEFAULT)
        #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 15)
        
        # Recognize the first and last characters, and correct the image
        first_char_coordinates, last_char_coordinates, number_of_characters = self.get_corners_of_text(image=unisize)
        # If the required conditions are passed do to correction, othrwise use the grayscale image
        if(first_char_coordinates is not None and number_of_characters > 4):
            # Binarize conversion
            ret3, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # Performing perspective transformation
            transformed_image = self.do_perspective_transform(image=thresh, text_box_coordinates=(first_char_coordinates, last_char_coordinates))
            # It adds some white edge to improve the effectiveness of OCR
            extended = self.pad_image(transformed_image, 0.1, color="white")
            # Dilate on the image (if you dilate an inverse image, you erode it)
            dilated = cv2.dilate(extended, np.ones((3, 3), np.uint8))
            # Open ot the image
            opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            #cv2.imshow('image', opening) # For debug
            #cv2.waitKey() # For debug
            return opening # image
        else:
            return gray_image
    

    def perscpective_correction(self, image, general_resize_factor):
        # It does not perform a complete perspective transformation (just a simple correction), 
        # but this step may be enough if the input image is horizontal enough.
        
        # The aspect ratio of the number plates must be uniform.
        scale_percent = 520/110 # percent of original size
        initial_height = image.shape[0]

        width = int((initial_height * scale_percent)*general_resize_factor)
        height = int(initial_height*general_resize_factor)
  
        horizontally_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

        return horizontally_resized # image
    

    def get_corners_of_text(self, image): # Karakter kép kiemelése a rendszám képéről
        # Search the characters in the image
        try:
            results = self.model_extract_char.predict(source=image, conf=0.5, verbose=False)
        except Exception as e:
            # If any problems occur
            print("Error: Character locating AI not working")
            print(e)
            return None, None, None
        
        # Sort the characters and return with their coordinates
        try:
            sorted_char_coordinates = sorted(results[0].boxes.xyxy, key=lambda x: x.cpu().numpy().astype(int)[0])
            number_of_characters = len(sorted_char_coordinates)
            first_char_coordinates = sorted_char_coordinates[0].cpu().numpy().astype(int)
            last_char_coordinates = sorted_char_coordinates[-1].cpu().numpy().astype(int)
        except Exception as e:
            print("Error: Could not give text bounding box coordinates")
            print(e)
            return None, None, None

        return first_char_coordinates, last_char_coordinates, number_of_characters
    
    
    def do_perspective_transform(self, image, text_box_coordinates):
        # Image skew correction with perspective transform
        src = np.float32([[text_box_coordinates[0][0],text_box_coordinates[0][1]], [text_box_coordinates[1][2],text_box_coordinates[1][1]],
                                [text_box_coordinates[0][0],text_box_coordinates[0][3]], [text_box_coordinates[1][2],text_box_coordinates[1][3]]])
        dst = np.float32([[0, 0], [440, 0],
                          [0, 110], [440, 110]])
        # Create the transformation matrix
        matrix = cv2.getPerspectiveTransform(src, dst)
        # Performing the transformation
        result = cv2.warpPerspective(image, matrix, (440, 110), borderMode=cv2.BORDER_CONSTANT, borderValue = [230, 230, 230])
        return result
        

    # It adds padding to improve OCRs working
    def pad_image(self, src, padding, color=None):
        top = int(padding * src.shape[0])
        bottom = top
        left = int(padding * src.shape[1])
        right = left
        if(color == None):
            dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_REPLICATE)
        elif(color == "white"):
            dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        elif(color == "black"):
            dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        return dst


