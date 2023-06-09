# Import libraries
import pytesseract

# We are using Tesseract OCR to detect a text in the the image
class image_to_text():

    def __init__(self):
        # Uncomment this if your Tesseract OCR is not added to environment variables
        #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        self.char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-' # Whitelist It will be filtered for these caracters

    # Using text recognition - Tesseract OCR
    def get_text(self, image):
        if(image is None): return "Text not recognized" #
        try:
            #text = pytesseract.image_to_string(image, config ='--psm 6')
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist='+self.char_whitelist)
        except Exception as e:
            print("Error: Text recognition OCR not working")
            print(e)
            return "Text not recognized"
        if(text is None or len(text) < 6 or len(text) > 10): # We know that the length of the number plate should be between 6 and 10 characters.
            print("Error: Text recognition OCR working, but returns invalid string")
            return "Text not recognized"
        return self.format_text(text) # string
    
    def format_text(self, text): # SZÖVEG FORMÁZÁSA, REGEX, STB IDE JÖHET
        try:
            # It removes the non uppercase and non alphabetical characters from the beginning of the string
            # for example: invalid characters or country codes
            while(text[0].isupper() is False or text[0].isalpha() is False and len(text) > 1):
                text = text[1:]
            
            # Ensure that the output string contains only numbers, letters and hyphens
            temp_text_1 = ''
            for i in range(len(text)):
                if(text[i].isnumeric() or (text[i].isalpha() and text[i].isupper()) or text[i]=='-'):
                    temp_text_1 += text[i]
            
            formatted_text = ''
            for i in range(len(temp_text_1)): # HA 3 BETŰ - 3 SZÁM
                # We assume that the number plate contains seven characters.
                # If the first three characters contains any number, we must replace it with a similar letter
                # If the last three characters contains any letter, we must replace it with a similar number
                if('-' in temp_text_1 and i > 3 and len(temp_text_1) == 7):
                    formatted_text += self.replace_alpha_with_num(temp_text_1[i])
                elif('-' in temp_text_1 and i < 3 and len(temp_text_1) == 7):
                    formatted_text += self.replace_num_with_alpha(temp_text_1[i])
                else:
                    formatted_text += temp_text_1[i]

            # If there is no hyphen between the letters and the numbers we put it there
            if(len(formatted_text) == 6): formatted_text = formatted_text[:3] + '-' + formatted_text[3:]
            
            return formatted_text # string
        except Exception as e:
            print("Warning: Text formatting error, raw text returned")
            print(e)
            return text
        

    # These functions are used to replace numbers and letters that look similar
    def replace_num_with_alpha(self, character):
        character = character.replace("5", "S")
        character = character.replace("0", "O")
        character = character.replace("7", "Z")
        character = character.replace("8", "B")
        character = character.replace("1", "I")
        return character # char
    
    def replace_alpha_with_num(self, character):
        character = character.replace("S", "5")
        character = character.replace("O", "0")
        character = character.replace("Z", "7")
        character = character.replace("B", "8")
        character = character.replace("I", "1")
        return character # char
