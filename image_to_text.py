import pytesseract

class image_to_text():

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    def get_text(self, image):
        text = pytesseract.image_to_string(image, config ='--psm 6')
        return self.format_text(text)
    
    def format_text(self, text):
        if(text is None): return "Not Recognized"
        try:
            formatted_text = ''
            for i in range(len(text)):
                if(text[i].isnumeric() or text[i].isalpha() or text[i]=='-'):
                    formatted_text += text[i]
            
            sliced_text = formatted_text[formatted_text.index('-')-3:formatted_text.index('-')+4]
            return sliced_text
        except:
            return "Not Recognized"