from gtts import gTTS
import random
from time import sleep
import os
import vlc

random.seed(42)
tts = gTTS(text='Hello World', lang='en')
FILENAME = '/tmp/temp' + str(random.randrange(1000000)) + '.mp3'
print("Writing mp3 to file: ", FILENAME)
tts.save(FILENAME)

p = vlc.MediaPlayer("file:///path/to/track.mp3")
p.play()
sleep(5)

os.remove(FILENAME)  # remove temperory file
