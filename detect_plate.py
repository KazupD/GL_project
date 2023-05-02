import cv2

class detect_plate():
    def __init__(self):
        pass

    def get_plate_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        blur = cv2.bilateralFilter(im_bw, 13, 15, 15)

        edged = cv2.Canny(blur, 30, 200) 

        cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:10]

        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(c) 
                new_img=image[y:y+h,x:x+w]
                
                return new_img
            
        return image
