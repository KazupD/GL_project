from ultralytics import YOLO
import cv2
import numpy as np

class isolate_character():
    def __init__(self):
        self.model = YOLO('yolo_custom_char_250.pt')

    
    def get_char_image(self, image): # Karakter kép kiemelése a rendszám képéről
        try: 
            results = self.model.predict(source=image, conf=0.5, verbose=False)
        except Exception as e:
            print("Error: Character locating AI not working")
            print(e)
            return image # Eredeti kép visszaadása
        try:
            extracted_char_imgs = []
            for result in results[0].boxes.xyxy:
                char_coordinates = result.cpu().numpy().astype(int)
                extracted_char_imgs.append(image[char_coordinates[1]:char_coordinates[3], char_coordinates[0]:char_coordinates[2]])
        except Exception as e:
            print("Error: Result converting not working")
            print(e)
            return image # Eredeti kép visszaadása

        return extracted_char_imgs
    
    def get_corners_of_text(self, image): # Karakter kép kiemelése a rendszám képéről
        try: 
            results = self.model.predict(source=image, conf=0.5, verbose=False)
        except Exception as e:
            print("Error: Character locating AI not working")
            print(e)
            return image # Eredeti kép visszaadása
        try:
            sorted_char_coordinates = sorted(results[0].boxes.xyxy, key=lambda x: x.cpu().numpy().astype(int)[0])
            first_char_coordinates = sorted_char_coordinates[0].cpu().numpy().astype(int)
            last_char_coordinates = sorted_char_coordinates[-1].cpu().numpy().astype(int)
        except Exception as e:
            print("Error: Could not give text bounding box coordinates")
            print(e)
            return image # Eredeti kép visszaadása

        return (first_char_coordinates, last_char_coordinates)
    

    def do_perspective_transform(self, image, text_box_coordinates):
        # ha lehetséges, hagyjon egy kis helyet, hogy a betűk ne a kép széléig érjenek
        # állítólag ez javítja az OCR esélyeit
        try:
            src = np.float32([[text_box_coordinates[0][0]-10,text_box_coordinates[0][1]-10], [text_box_coordinates[1][2]+10,text_box_coordinates[1][1]-10],
                            [text_box_coordinates[0][0]-10,text_box_coordinates[0][3]+10], [text_box_coordinates[1][2]+10,text_box_coordinates[1][3]+10]])
            dst = np.float32([[0, 0], [440, 0],
                            [0, 110], [440, 110]])
            matrix = cv2.getPerspectiveTransform(src, dst)
            result = cv2.warpPerspective(image, matrix, (440, 110), borderMode=cv2.BORDER_CONSTANT, borderValue = [230, 230, 230])
        except:
            try:
                src = np.float32([[text_box_coordinates[0][0],text_box_coordinates[0][1]], [text_box_coordinates[1][2],text_box_coordinates[1][1]],
                                [text_box_coordinates[0][0],text_box_coordinates[0][3]], [text_box_coordinates[1][2],text_box_coordinates[1][3]]])
                dst = np.float32([[0, 0], [440, 0],
                                [0, 110], [440, 110]])
                matrix = cv2.getPerspectiveTransform(src, dst)
                result = cv2.warpPerspective(image, matrix, (440, 110), borderMode=cv2.BORDER_CONSTANT, borderValue = [230, 230, 230])
            except:
                result = image
        # cv2.imshow("Persp", result)
        # cv2.waitKey(0)