# Import libraries
import csv
import numpy as np
import urllib.request
import cv2


class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

class fetch_car():

    def __init__(self, db_path = "database/db.csv", db_delimiter = ";"):
        self.db_path = db_path
        self.db_delimiter = db_delimiter
        self.opener = AppURLopener()

    #
    def set_dbpath(self, db_path):
        self.db_path = db_path

    def load_by_numberplate(self, numberplate):
        with open(self.db_path, encoding="utf8") as db:
            reader=csv.reader(db, delimiter=self.db_delimiter)
            
            #car_data structure: 0: numberplate, 1: numberplate with car type, 2: rendered numberplate, 3: large image, 4: small image
            car_data = [r for idx, r in enumerate(reader) if r[0] == numberplate][0]

            images = []
            for i in range(len(car_data)):
                if(car_data[i].startswith("http")):
                    try:
                        req = self.opener.open(car_data[i])
                        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
                        images.append(cv2.imdecode(img, -1)) # It adds the image to the image list
                    except Exception as e:
                        print("Failed Request:" + e)
                        return None
                    
            return images # image list 3

    # Read the csv file and return 
    def load_by_index(self, index):
        with open(self.db_path, encoding="utf8") as db:
            reader=csv.reader(db, delimiter=self.db_delimiter)
            car_data = [r for idx, r in enumerate(reader) if idx == index][0]

            images = []
            for i in range(len(car_data)):
                if(car_data[i].startswith("http")):
                    try:
                        req = self.opener.open(car_data[i])
                        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
                        images.append(cv2.imdecode(img, -1))
                    except Exception as e:
                        print("Failed Request:" + e)
                        return None
                    
            return images # image list 3
    
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
        
    # 
    def load_image_by_url(self, url):
        try:
            req = self.opener.open(url)
            img = np.asarray(bytearray(req.read()), dtype=np.uint8)
            return cv2.imdecode(img, -1) # image
        except Exception as e:
            print("Failed Request:" + e)
            return None
