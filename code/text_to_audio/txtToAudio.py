from gtts import gTTS #google Text-To-Speech library
import os 
from text_to_audio.spell_checker import check_spelling


def txtToAud(txt):  
	# 'txt' is comverted into audio
	# Language of preference
	language = 'en'
	txt = check_spelling(txt)

	myobj = gTTS(text=txt, lang=language, slow=False) 

	#save myobj as an mp3 file  
	myobj.save("welcome.mp3") 
	  
	#Projects audio
	os.system("mpg321 welcome.mp3")

#txtToAud("Hello, persin in my houxe ")