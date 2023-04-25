from fetch_car import fetch_car
import cv2


def main():
    car_loader = fetch_car()

    images = car_loader.load_by_numberplate('PLG-778')

    #images = car_loader.load_by_index(4)

    #print(car_loader.get_index_by_numberplate('FAU-023'))

    #print(car_loader.get_numberplate_by_index(3))

    for i in range(len(images)):
        cv2.imshow("image", images[i])
        cv2.waitKey(0)
    

if __name__ == "__main__":
    main()