import pytesseract

class image_to_text():

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        self.char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-'

    def get_text(self, image): # SZÖVEG FELISMERŐ OCR HASZNÁLATA, MÁS NE LEGYEN ITT
        try:
            #text = pytesseract.image_to_string(image, config ='--psm 6')

            text = pytesseract.image_to_string(image, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist='+self.char_whitelist)
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
                    formatted_text+=text[i]

            return formatted_text
        except:
            print("Warning: Text formatting error, raw text returned")
            return text
        
    def replace_num_with_alpha(self, character):
        new_character = character.replace("5", "S")
        new_character = character.replace("0", "O")
        new_character = character.replace("7", "Z")
        new_character = character.replace("8", "B")
        new_character = character.replace("1", "I")
        return new_character
    
    def replace_alpha_with_num(self, character):
        new_character = character.replace("S", "5")
        new_character = character.replace("O", "0")
        new_character = character.replace("Z", "7")
        new_character = character.replace("B", "8")
        new_character = character.replace("I", "1")
        return new_character
