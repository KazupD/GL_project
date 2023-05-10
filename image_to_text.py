import pytesseract

class image_to_text():

    def __init__(self):
        #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        self.char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-'

    def get_text(self, image): # SZÖVEG FELISMERŐ OCR HASZNÁLATA, MÁS NE LEGYEN ITT
        try:
            #text = pytesseract.image_to_string(image, config ='--psm 6')
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist='+self.char_whitelist)
        except Exception as e:
            print("Error: Text recognition OCR not working")
            print(e)
            return "Text not recognized"
        if(text is None or len(text) < 2):
            print("Error: Text recognition OCR working, but returns invalid string")
            return "Text not recognized"
        return self.format_text(text)
    
    def get_chars(self, images):
        chars = []
        for image in images:
            try: # A char egy tömb
                char = pytesseract.image_to_string(image, lang='eng', config='--psm 10 --oem 1 -c tessedit_char_whitelist='+self.char_whitelist)
            except Exception as e:
                print("Error: Character recognition OCR not working")
                print(e)
                return "Text not recognized"
            chars.append(char)
        print("Recognized characters: ",chars)


    
    def format_text(self, text): # SZÖVEG FORMÁZÁSA, REGEX, STB IDE JÖHET
        try:
            while(text[0].isupper() is False or text[0].isalpha() is False):  text = text[1:] # Zászló, EU jel, kis H betű leszedése
            temp_text_1 = ''
            for i in range(len(text)):
                if(text[i].isnumeric() or (text[i].isalpha() and text[i].isupper()) or text[i]=='-'):
                    temp_text_1 += text[i]
            
            formatted_text = ''
            for i in range(len(temp_text_1)): # HA 3 BETŰ - 3 SZÁM
                if('-' in temp_text_1 and i > 3 and len(temp_text_1) == 7):
                    formatted_text += self.replace_alpha_with_num(temp_text_1[i])
                elif('-' in temp_text_1 and i < 3 and len(temp_text_1) == 7):
                    formatted_text += self.replace_num_with_alpha(temp_text_1[i])
                else:
                    formatted_text += temp_text_1[i]
            
            return formatted_text
        except Exception as e:
            print("Warning: Text formatting error, raw text returned")
            print(e)
            return text
        
    def replace_num_with_alpha(self, character):
        character = character.replace("5", "S")
        character = character.replace("0", "O")
        character = character.replace("7", "Z")
        character = character.replace("8", "B")
        character = character.replace("1", "I")
        return character
    
    def replace_alpha_with_num(self, character):
        character = character.replace("S", "5")
        character = character.replace("O", "0")
        character = character.replace("Z", "7")
        character = character.replace("B", "8")
        character = character.replace("I", "1")
        return character
