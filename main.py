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

    images = img_fetch.load_by_numberplate('MRR-889')
    print(img_to_txt.get_text(images[0])) 
    plate = img_plate_detect.get_plate_image(images[1])
    cv2.imshow("Numberplate", images[1])
    cv2.waitKey(0)
    cv2.imshow("Numberplate", plate)
    print(img_to_txt.get_text(plate))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #test_on_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt)

    # MRR-889; FFX-966 például fixen működik

if __name__ == "__main__":
    main()