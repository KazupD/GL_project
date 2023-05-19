# Import libraries
import csv
import numpy as np
import urllib.request
import cv2

# Library describtion
# This library is used to read the appropriate formatted database to retrieve images from the Internet.
# It contains functions that can be used to request images for debugging or benchmarking, or for use in real applications.
# load_ functions are used to request the images
# get_ functions are used to extract information(s) from the database by given data

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

class fetch_car():

    def __init__(self, db_path = "database/db.csv", db_delimiter = ";"):
        self.db_path = db_path # Set the database path in constructor
        self.db_delimiter = db_delimiter # Set the delimiter
        self.opener = AppURLopener()

    # Set the database path by setter function
    def set_dbpath(self, db_path):
        self.db_path = db_path 

    
    # This function combines the number plate and index query functions.
    def load(self, numberplate = None, index = None, url = None):
        if url is not None:
            return self.load_image_by_url(url)
        if numberplate is not None or index is not None:
            with open(self.db_path, encoding="utf8") as db:
                reader=csv.reader(db, delimiter=self.db_delimiter)
                #car_data structure: 0: numberplate, 1: numberplate with car type, 2: rendered numberplate, 3: large image, 4: small image
                car_data = None
                if(numberplate is not None):
                    car_data = [r for idx, r in enumerate(reader) if r[0] == numberplate][0]
                elif(index is not None):
                    car_data = [r for idx, r in enumerate(reader) if idx == index][0]
                else:
                    return None
                    
                images = [] # Create an array to store the downloaded images
                for i in range(len(car_data)):
                    if(car_data[i].startswith("http")): # Filter by starting string
                        images.append(self.load_image_by_url(car_data[i]))
                return images # image list 3
            


    # It returns the images where the number plate is.
    def load_by_numberplate(self, numberplate):
        with open(self.db_path, encoding="utf8") as db:
            reader=csv.reader(db, delimiter=self.db_delimiter)
            
            #car_data structure: 0: numberplate, 1: numberplate with car type, 2: rendered numberplate, 3: large image, 4: small image
            car_data = [r for idx, r in enumerate(reader) if r[0] == numberplate][0]

            images = [] # Create an array to store the downloaded images
            for i in range(len(car_data)):
                if(car_data[i].startswith("http")): # Filter by starting string
                    try:
                        req = self.opener.open(car_data[i]) # It makes an http request to get get the pictures
                        img = np.asarray(bytearray(req.read()), dtype=np.uint8) # It converts the image to numpy array
                        images.append(cv2.imdecode(img, -1)) # It decodes and adds the image to the image array
                    except Exception as e:
                        print(e)
                        return None
                    
            return images # image list 3


    # It returns the images associated with the row with the given index
    def load_by_index(self, index):
        with open(self.db_path, encoding="utf8") as db:
            reader=csv.reader(db, delimiter=self.db_delimiter)

            #car_data structure: 0: numberplate, 1: numberplate with car type, 2: rendered numberplate, 3: large image, 4: small image
            car_data = [r for idx, r in enumerate(reader) if idx == index][0]

            images = [] # Create an array to store the downloaded images
            for i in range(len(car_data)):
                if(car_data[i].startswith("http")): # Filter by starting string
                    try:
                        req = self.opener.open(car_data[i]) # It makes an http request to get get the pictures
                        img = np.asarray(bytearray(req.read()), dtype=np.uint8) # It converts the image to numpy array
                        images.append(cv2.imdecode(img, -1)) # It decodes and adds the image to the image array
                    except Exception as e:
                        print(e)
                        return None
                    
            return images # image list 3
    
    # It returns with the image given with the url
    def load_image_by_url(self, url):
        try:
            req = self.opener.open(url) # It makes an http request to get get the pictures
            img = np.asarray(bytearray(req.read()), dtype=np.uint8) # It converts the image to numpy array
            return cv2.imdecode(img, -1) # It decodes and adds the image to the image array
        except Exception as e:
            print(e)
            return None

    # It returns with the index by the numberplate
    def get_index_by_numberplate(self, numberplate):
        with open(self.db_path, encoding="utf8") as db:
            reader=csv.reader(db, delimiter=self.db_delimiter)
            for r in enumerate(reader):
                if(r[1][0] == numberplate): return r[0] # int
            return -1

    # It returns with the numberplate string
    def get_numberplate_by_index(self, index):
        with open(self.db_path, encoding="utf8") as db:
            reader=csv.reader(db, delimiter=self.db_delimiter)
            row = [r for idx, r in enumerate(reader) if idx == index]
            return row[0][0] # string
        

