from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
from ultralytics import YOLO
import cv2


img_fetch = fetch_car()

images = img_fetch.load_by_numberplate('EFR-179')

# Initialisaiton: Load trained model
model = YOLO('trained_90.pt')

def locate_plate(model, image):
    # Run prediction algorithm
    result = model.predict(source=image, conf=0.5, verbose=False)

    # For testing
    if len(result) == 0:
        print("Error: Could not find licence plate on image") # Szükség esetén jól jöhet, ha megmondja, melyik képen hasalt el

    if len(result) > 1:
        print("Error: More than one licence plate detected on image") # Itt is szükség esetén jól jöhet, ha megmondja, melyik képen hasalt el

    # Return bounding box coordinates of detected license plate as a numpy array
    return result[0].boxes.xyxy[0].cpu().numpy()

# Hogy lásd, mi az output
print(locate_plate(model, images[1]))