from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
import cv2

def download_picturte_for_ocr_training(img_fetch, img_plate_detect, img_to_text):
    test_numbers = 5
    for i in range(test_numbers):
        images = img_fetch.load_by_index(i)
        # A csak rendszámot tartalmazó kép : images[0]
        plate = img_plate_detect.get_plate_image(images[1])
        # gray_image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        #(thresh, plate_bin) = cv2.threshold(plate, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        original_text = img_fetch.get_numberplate_by_index(i)

        if plate.shape == (1080, 1440, 3):
            print(f"Could not detect plate on image {i}. Skipping")
            continue

        cv2.imwrite(f"/media/barni/ostree/GL_data/OCR_training/test_ocr/test_{i}.png", plate)
        cv2.imshow("asd", plate)
        cv2.waitKey(0)
        with open(f"/media/barni/ostree/GL_data/OCR_training/test_ocr/test_{i}.gt.txt", 'w') as f:
            f.write(original_text)

        print(f"Downloaded image {i}")

def download_render_for_ocr_training(img_fetch, img_plate_detect, img_to_text):
    test_numbers = 500
    for i in range(test_numbers):
        images = img_fetch.load_by_index(i)
        # A csak rendszámot tartalmazó kép : images[0]
        plate = img_plate_detect.get_plate_image(images[0])
        cropped_image = plate[6:44, 33:227 , :]
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        (thresh, plate_bin) = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        original_text = img_fetch.get_numberplate_by_index(i)

        if plate.shape == (1080, 1440, 3):
            print(f"Could not detect plate on image {i}. Skipping")
            continue

        cv2.imwrite(f"/media/barni/ostree/GL_data/OCR_training/data/HUNPlate-ground-truth/training_{i}.tif", plate_bin)
        with open(f"/media/barni/ostree/GL_data/OCR_training/data/HUNPlate-ground-truth/training_{i}.gt.txt", 'w') as f:
            f.write(original_text)

        print(f"Downloaded image {i}")

def download_image_for_testing(img_fetch, img_plate_detect):
    start_index = 0
    test_number = 300
    for i in range(start_index, start_index+test_number):
        print(i)
        original_images = img_fetch.load_by_index(i)
        plate = img_plate_detect.get_plate_image(original_images[1])
        original_text = img_fetch.get_numberplate_by_index(i)
        cv2.imwrite(f"input/{original_text}_{i}.png", plate)


def download_images():
    img_fetch = fetch_car()
    img_plate_detect = detect_plate()
    img_to_txt = image_to_text()

    download_image_for_testing(img_fetch=img_fetch, img_plate_detect=img_plate_detect)

if __name__ == "__main__":
    download_images()