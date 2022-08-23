import pyttsx3
import sys

def text_to_speech(text):
    text_speech = pyttsx3.init()
    text_speech.say(text)
    text_speech.runAndWait()

if __name__ == "__main__":
    text = sys.argv[1]
    text_to_speech(text)