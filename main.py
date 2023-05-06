from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
import cv2

def test_on_database(img_fetch, img_plate_detect, img_to_text):
    test_numbers = 50
    success_numbers = 0
    for i in range(test_numbers):
        images = img_fetch.load_by_index(i)
        # A csak rendszámot tartalmazó kép : images[0]
        plate = img_plate_detect.get_plate_image(images[1])
        original_text = img_fetch.get_numberplate_by_index(i)
        detected_text = img_to_text.get_text(plate)
        #print(original_text)
        #print(detected_text)
        print(i)
        if(original_text == detected_text): success_numbers+=1

    print("Success rate : " + str((success_numbers/test_numbers)*100) + "%")


def main():
    img_fetch = fetch_car()
    img_plate_detect = detect_plate()
    img_to_txt = image_to_text()

    '''images = img_fetch.load_by_numberplate('MRR-889')
    cv2.imshow("Car", images[2])
    plate = img_plate_detect.get_plate_image(images[1])
    print(img_to_txt.get_text(plate))
    cv2.imshow("Numberplate", plate)
    cv2.waitKey(0)'''
    

    test_on_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt)

    # Jók: MRR-889; FFX-966; LLP-676; XYD-635; FCR-841; LPY-437; FAU-023
    # Rosszak: JTZ-465; GFX-767; PWR-923; HON-804; DEL-011; HFP-620; GTH-057; AYA-599; NWX-474; RFW-499


if __name__ == "__main__":
    main()