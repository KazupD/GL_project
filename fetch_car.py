import csv
import numpy as np
import urllib.request
import cv2

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

class fetch_car():

    def __init__(self):
        self.opener = AppURLopener()

    def load_by_numberplate(self, numberplate):
        with open('database/HF_train_database.csv', encoding="utf8") as db:
            reader=csv.reader(db, delimiter=";")

            row = [r for idx, r in enumerate(reader) if r[0] == numberplate]
            car_data = row[0]

            images = []
            for i in range(len(car_data)):
                if(car_data[i].startswith("http")):
                    
                    try:
                        req = self.opener.open(car_data[i])
                        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
                        images.append(cv2.imdecode(img, -1))
                    except Exception as e:
                        print(e)
                        print("Failed Request")
                        return None
                    
            return images

    def load_by_index(self, index):
        with open('database/HF_train_database.csv', encoding="utf8") as db:
            reader=csv.reader(db, delimiter=";")

            row = [r for idx, r in enumerate(reader) if idx == index]
            car_data = row[0]

            images = []
            for i in range(len(car_data)):
                if(car_data[i].startswith("http")):
                    
                    try:
                        req = self.opener.open(car_data[i])
                        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
                        images.append(cv2.imdecode(img, -1))
                    except Exception as e:
                        print(e)
                        print("Failed Request")
                        return None
                    
            return images
        
    def get_index_by_numberplate(self, numberplate):
        with open('database/HF_train_database.csv', encoding="utf8") as db:
            reader=csv.reader(db, delimiter=";")

            for r in enumerate(reader):
                if(r[1][0] == numberplate): return r[0]
            
            return -1

    def get_numberplate_by_index(self, index):
        with open('database/HF_train_database.csv', encoding="utf8") as db:
            reader=csv.reader(db, delimiter=";")

            row = [r for idx, r in enumerate(reader) if idx == index]
            return row[0][0]