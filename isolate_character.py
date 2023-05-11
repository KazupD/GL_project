from ultralytics import YOLO

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