# Import libraries
from fetch_car import fetch_car
from image_to_text import image_to_text
from detect_plate import detect_plate
import cv2
import os
import csv
import time
import sys

#### Program paramatres
debug_output_path = "output" # Image output folder
if not os.path.exists(debug_output_path): os.makedirs(debug_output_path) # Create path if not exist
delimiter = ";" # CSV Database delimiter caracter
# We defined two different type of database
# The benchmark version contains tagged images, and the other contains only images
benchmark_CSVdatabase_path = "database/HF_train_database.csv" # Labelled images
CSVdatabase_path = 'database/HF_final_database_beta.csv' # Images what you want to evaluate

# This function can read and evaluate the test database
# The result of the evaluated image will be printed to the console
# This function can be used to benchmark our code
def test_on_database(img_fetch, img_plate_detect, img_to_text):
    global benchmark_CSVdatabase_path
    # Testing database parameters
    start_index = 500 # This is the index of the first evaluated row
    test_number = 10 # This number of images will be evaluated
    # Statistic variables
    success_numbers = 0 # How many puctures were corretly evaluated?
    start_time = time.time() # Start a timer (get the current time)
    not_evaluation_errors = 0 # The number of errors during database processing

    # Iterate on the database file
    for i in range(start_index, start_index+test_number):
        # Load the picture from the database by the line index
        images = img_fetch.load_by_index(i)
        if(images is not None): # If the HTTP connection works correctly
            # Get and evaluate the image
            plate = img_plate_detect.get_plate_image(images[1])
            original_text = img_fetch.get_numberplate_by_index(i)
            detected_text = img_to_text.get_text(plate)

            # Display the current detection status
            # Display the image number, original text and the detected test to the console
            print("Image number:" + str(i))
            print(original_text)
            print(detected_text)
            is_ok = bool(original_text == detected_text)
            if(is_ok): success_numbers+=1
            #cv2.imwrite(path +"/"+ str(i) + "_"+ str(is_ok) + ".jpg", plate)
            print("Success? -> " +  str(is_ok))
            print("-------------------------------")

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
def test_on_final_database(img_fetch, img_plate_detect, img_to_text):
    print(CSVdatabase_path)
    file_to_complete_data = []
    with open(CSVdatabase_path, encoding="utf8") as db:
        # Read files
        file=csv.reader(db, delimiter=delimiter)
        rowindex = 0
        for row in file:
            numberplate_value = '' # Initialize variable
            image = img_fetch.load_image_by_url(row[1]) # GET the image by url
            if(image is not None): # Evaluate image if there is no db error
                plate = img_plate_detect.get_plate_image(image)
                numberplate_value = img_to_text.get_text(plate)
            file_to_complete_data.append([numberplate_value, row[1], row[2]]) # Add a row to a
            print(rowindex)
            rowindex+=1
        
    with open('database/CirmosCicak.csv', 'w', encoding="utf8", newline ='') as db:  
        write = csv.writer(db, delimiter=delimiter)
        write.writerows(file_to_complete_data)

def evaluate_in_custom_directory(img_plate_detect, img_to_text):
    file_to_complete_data = []
    if len(sys.argv) == 2:
        dir = sys.argv[1]
    else:
        print("Usage: python main.py folder-to-evaluate")
    
    files = os.listdir(dir)
    test_number = len(files)
    start_time = time.time()

    for index, file in enumerate(files):
        numberplate_value = '' # Initialize variable
        image = cv2.imread(dir+file)
        if(image is not None): # Evaluate image if there is no db error
            plate = img_plate_detect.get_plate_image(image)
            numberplate_value = img_to_text.get_text(plate)
        file_to_complete_data.append([file, numberplate_value])
        print(index)

    with open('database/CirmosCicak.csv', 'w', encoding="utf8", newline ='') as db:  
        write = csv.writer(db, delimiter=delimiter)
        write.writerows(file_to_complete_data)

    finish_time = time.time()
    total_time = round(finish_time-start_time, 3)
    average_time = round(total_time/test_number, 3)
    print("Total time: " + str(total_time) + " s")
    print("Average time: " + str(average_time) + " s")

def main():
    # Create object 
    img_fetch = fetch_car()
    img_plate_detect = detect_plate()
    img_to_txt = image_to_text()

    #Mode selector - Choose the way you want to use the app
    # Mode 0: - evaluate images (real-life application)
    # Mode 1: - evaluate only one image (Debug application)
    # Mode 2: - evaluate the test database (benchmark application)
    mode = 3
    if (mode == 0):
        img_fetch.set_dbpath(CSVdatabase_path) # Set the path to database
        test_on_final_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt)
    elif (mode == 1):
        img_fetch.set_dbpath(benchmark_CSVdatabase_path)
        images = img_fetch.load_by_numberplate("AYA-599")
        cv2.imshow("Car", images[2])
        plate = img_plate_detect.get_plate_image(images[1])
        print(img_to_txt.get_text(plate))
        cv2.imshow("Numberplate", plate)
        cv2.waitKey(0)
    elif (mode == 2):
        img_fetch.set_dbpath(benchmark_CSVdatabase_path)
        test_on_database(img_fetch=img_fetch, img_plate_detect=img_plate_detect, img_to_text=img_to_txt)
    elif (mode == 3):
        evaluate_in_custom_directory(img_plate_detect=img_plate_detect, img_to_text=img_to_txt)

if __name__ == "__main__":
    main()