
from gtts import gTTS
text=""

speech=gTTS(text,'en','slow')

speech.save("hello.mp3")
