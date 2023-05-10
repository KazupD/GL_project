from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
from isolate_character import isolate_character
import cv2

#def test_on_database(img_fetch, img_plate_detect, img_to_text, isolate_char):
def test_on_database(img_fetch, img_to_text, isolate_char):
    start_index = 100
    test_number = 1
    success_numbers = 0
    for i in range(start_index, start_index+test_number):
        images = img_fetch.load_by_index(i)
        #plate = img_plate_detect.get_plate_image(images[1])
        plate = cv2.imread("output/2_True.jpg")
        #cv2.imshow("plate", plate)
        #cv2.waitKey(0)
        original_text = img_fetch.get_numberplate_by_index(i)
        character_images = isolate_char.get_char_image(plate)
        img_to_text.get_chars(images=character_images)
        #img_to_text.get_chars(character_images)
        #detected_text = img_to_text.get_text(plate)
        #cv2.imwrite('image_buffer/plate'+str(i)+'_'+detected_text+'.png', plate)
        
        #print(i)
        #if(original_text == detected_text): success_numbers+=1

    print("Success rate : " + str((success_numbers/test_number)*100) + "%")


def main():
    img_fetch = fetch_car()
    #img_plate_detect = detect_plate()
    img_to_txt = image_to_text()
    isolate_char = isolate_character()

    '''images = img_fetch.load_by_index(520)
    cv2.imshow("Car", images[2])
    plate = img_plate_detect.get_plate_image(images[1])
    print(img_to_txt.get_text(plate))
    cv2.imshow("Numberplate", plate)
    cv2.waitKey(0)'''
    

    #test_on_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt, isolate_char=isolate_char)
    test_on_database(img_fetch=img_fetch, img_to_text=img_to_txt, isolate_char=isolate_char)

    # Jók: MRR-889; FFX-966; LLP-676; XYD-635; FCR-841; LPY-437; FAU-023; NWX-474
    # Rosszak: JTZ-465; GFX-767; PWR-923; HON-804; DEL-011; HFP-620; GTH-057; AYA-599; RFW-499


if __name__ == "__main__":
    main()