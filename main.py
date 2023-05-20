# Import libraries
from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
import cv2
import os
import csv
import time

#### Program paramaters
delimiter = ";" # CSV Database delimiter caracter

# This function can read and evaluate the test database
# The result of the evaluated image will be printed to the console
# This function can be used to benchmark our code
def test_on_database(img_fetch, img_plate_detect, img_to_text, start_index = 0, test_number = 100, log = False, debug_output_path = "debug.csv"):
    # Statistic variables
    success_numbers = 0 # How many puctures were corretly evaluated?
    start_time = time.time() # Start a timer (get the current time)

    # Iterate on the database file
    for i in range(start_index, start_index+test_number):
        images = img_fetch.load_by_index(i) # Load the picture from the internet by the databae line index
        if(images is  None): continue # If the HTTP connection does not work correctly

        plate = img_plate_detect.get_plate_image(images[1]) # Get and evaluate the image
        original_text = img_fetch.get_numberplate_by_index(i)
        detected_text = img_to_text.get_text(plate)

        # Display the current detection status
        # Display the image number, original text and the detected test to the console
        is_ok = bool(original_text == detected_text)
        if(is_ok): success_numbers+=1
        print("Image number:" + str(i) + "\r\nOriginal:   " + original_text + "\r\nRecognized: " + detected_text + "\r\nSuccess? ->" +  str(is_ok) + "\r\n" + "-" * 10)
        if log:
            try:
                with open(f'{debug_output_path}/log.csv', 'a', newline='') as f_object:
                    writer_object = csv.writer(f_object)
                    row = [i, original_text, detected_text, str(is_ok)]
                    writer_object.writerow(row)
                    f_object.close()
                if is_ok is False:
                    fail_images_path = f"{debug_output_path}/fail"
                    if not os.path.exists(fail_images_path): os.makedirs(fail_images_path) # Create path if not exist
                    cv2.imwrite(f'{fail_images_path}/i{i}_{detected_text}.jpg', plate)
            except:
                print("Log error")    

        #cv2.imwrite(path +"/"+ str(i) + "_"+ str(is_ok) + ".jpg", plate)

    # Display the statistical datas
    print("Tested on : " +  str(test_number) + " images")
    print("Success rate : " + str((success_numbers/test_number)*100) + "%")
    finish_time = time.time()
    total_time = round(finish_time-start_time, 3)
    average_time = round(total_time/test_number, 3)
    print("Total time: " + str(total_time) + " s")
    print("Average time: " + str(average_time) + " s")


# The result of the evaluated image will be written to a CSV file
# This function can be used to detect and evaluate numberplates in the images
def test_on_final_database(img_fetch, img_plate_detect, img_to_text, database, database_out):
    file_to_complete_data = []
    with open(database, encoding="utf8") as db:
        # Read files
        file=csv.reader(db, delimiter=delimiter)
        rowindex = 0
        if(os.path.isfile(database_out)):
            if(os.stat("database/CirmosCicak.csv").st_size > 0):
                with open(database_out, 'a', newline='') as f_object:
                    f_object.write("\r\n")
                    f_object.close()

        for row in file:
            numberplate_value = '' # Initialize variable
            try:
                image = img_fetch.load_image_by_url(row[1]) # GET the image by url
                if(image is not None): # Evaluate image if there is no db error
                    plate = img_plate_detect.get_plate_image(image)
                    numberplate_value = img_to_text.get_text(plate)
                file_to_complete_data = [numberplate_value, row[1], row[2]] # Add a row to a
                with open(database_out, 'a', newline='') as f_object:
                    writer_object = csv.writer(f_object, delimiter=delimiter)
                    writer_object.writerow(file_to_complete_data)
                    f_object.close()
                print(f"{rowindex}. -> {numberplate_value}")
            except Exception as e:
                print(e)
            rowindex+=1
        
    #with open(database_out, 'w', encoding="utf8", newline ='') as db:  
    #    write = csv.writer(db, delimiter=delimiter)
    #    write.writerows(file_to_complete_data)

def test_debug(img_fetch, img_plate_detect, img_to_text, numberplate):
    images = img_fetch.load_by_numberplate(numberplate)
    cv2.imshow("Car", images[2])
    plate = img_plate_detect.get_plate_image(images[1])
    print(img_to_text.get_text(plate))
    cv2.imshow("Numberplate", plate)
    cv2.waitKey(0)



def main():
    # Create object 
    img_f = fetch_car()
    img_pd = detect_plate()
    img2txt = image_to_text()

    debug_output_path = "output" # Image output folder
    if not os.path.exists(debug_output_path): os.makedirs(debug_output_path) # Create path if not exist


    # We defined two different type of database
    # The benchmark version contains tagged images, and the other contains only images
    CSV_db_train = "database/HF_train_database.csv" # Labelled images
    CSV_db_app = 'database/HF_final_database_beta.csv' # Images what you want to evaluate
    CSV_db_OUT = 'database/CirmosCicak.csv' # It stores the generated datas

    #Mode selector - Choose the way you want to use the app
    # Mode 'REAL': - evaluate images (real-life application)
    # Mode 'DEBUG': - evaluate only one image (Debug application)
    # Mode 'BENCHMARK': - evaluate the test database (benchmark application)
    mode = 'BENCHMARK'
    debug_numberplate = "KYU-882"

    if (mode == 'REAL'):
        img_f.set_dbpath(CSV_db_app) # Set the path to database
        test_on_final_database(img_fetch=img_f, img_plate_detect=img_pd, img_to_text=img2txt, database = CSV_db_app, database_out = CSV_db_OUT)
    elif (mode == 'DEBUG'):
        img_f.set_dbpath(CSV_db_train)
        test_debug(img_fetch=img_f, img_plate_detect=img_pd, img_to_text=img2txt, numberplate=debug_numberplate)
    elif (mode == 'BENCHMARK'):
        img_f.set_dbpath(CSV_db_train)
        test_on_database(start_index = 0, test_number = 30, log = True, img_fetch=img_f, img_plate_detect=img_pd, img_to_text=img2txt, debug_output_path = debug_output_path)
    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()