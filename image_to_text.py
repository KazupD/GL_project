import pytesseract

class image_to_text():

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    def get_text(self, image): # SZÖVEG FELISMERŐ OCR HASZNÁLATA, MÁS NE LEGYEN ITT
        try:
            text = pytesseract.image_to_string(image, config ='--psm 6')
        except:
            print("Error: Text recognition OCR not working")
            return "Text not recognized"
        if(text is None or len(text) < 2):
            print("Error: Text recognition OCR working, but returns invalid string")
            return "Text not recognized"
        return self.format_text(text)
    
    def format_text(self, text): # SZÖVEG FORMÁZÁSA, REGEX, STB IDE JÖHET
        try:
            while(text[0].isupper() is False or text[0].isalpha() is False):  text = text[1:] # Zászló, EU jel, kis H betű leszedése
            formatted_text = ''
            for i in range(len(text)):
                if(text[i].isnumeric() or (text[i].isalpha() and text[i].isupper()) or text[i]=='-'):
                    formatted_text += text[i]
            
            #sliced_text = formatted_text[formatted_text.index('-')-3:formatted_text.index('-')+4]
            return formatted_text
        except:
            print("Warning: Text formatting error, raw text returned")
            return text