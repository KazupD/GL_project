from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
import cv2
import os
import csv
import time

path = "output"
if not os.path.exists(path): os.makedirs(path)

def test_on_database(img_fetch, img_plate_detect, img_to_text):
    start_index = 500
    test_number = 200
    success_numbers = 0
    start_time = time.time()
    for i in range(start_index, start_index+test_number):
        images = img_fetch.load_by_index(i)
        if(images is not None): # Ha nincs HTTP error
            plate = img_plate_detect.get_plate_image(images[1])
            original_text = img_fetch.get_numberplate_by_index(i)
            detected_text = img_to_text.get_text(plate)
            print("Image number:" + str(i))
            print(original_text)
            print(detected_text)
            is_ok = bool(original_text == detected_text)
            if(is_ok): success_numbers+=1
            #cv2.imwrite(path +"/"+ str(i) + "_"+ str(is_ok) + ".jpg", plate)
            print("Success? -> " +  str(is_ok))
            print("-------------------------------")


    print("Tested on : " +  str(test_number) + " images")
    print("Success rate : " + str((success_numbers/test_number)*100) + "%")
    finish_time = time.time()
    total_time = round(finish_time-start_time, 3)
    average_time = round(total_time/test_number, 3)
    print("Total time: " + str(total_time) + " s")
    print("Average time: " + str(average_time) + " s")


def test_on_final_database(img_fetch, img_plate_detect, img_to_text):
    file_to_complete_data = []
    with open('database/HF_final_database_beta.csv', encoding="utf8") as db:
        file=csv.reader(db, delimiter=";")
        rowindex = 0
        for row in file:
            numberplate_value = ''
            image = img_fetch.load_image_by_url(row[1])
            if(image is not None):
                plate = img_plate_detect.get_plate_image(image)
                numberplate_value = img_to_text.get_text(plate)
            file_to_complete_data.append([numberplate_value, row[1], row[2]])
            print(rowindex)
            rowindex+=1
        
    with open('database/CirmosCicak.csv', 'w', encoding="utf8", newline ='') as db:  
        write = csv.writer(db, delimiter=";")
        write.writerows(file_to_complete_data)


def main():
    img_fetch = fetch_car()
    img_plate_detect = detect_plate()
    img_to_txt = image_to_text()
    

    test_on_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt)
    #test_on_final_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt)


if __name__ == "__main__":
    main()