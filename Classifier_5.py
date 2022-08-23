from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from load_dataset_2 import extract_face
from face_embedding_4 import get_embedding
from keras.models import load_model

# load faces
data = load('D:/venv/dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('D:/venv/faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
model = load_model('D:/venv/facenet_keras.h5')
# new added code from aditya
image=extract_face('imageframe.jpg')
image=get_embedding(model,image)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# Take input image 
# input_img = 'D:/venv/train/ben_afflek/httpcsvkmeuaeccjpg.jpg'
# load_face = load_faces(input_img)
# embedded_face = get_embedding(load_face)

# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# samples = expand_dims(embedded_face, axis=0)
# yhat_class = model.predict(samples)
# yhat_prob = model.predict_proba(samples)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)

pyplot.title(title)
pyplot.show()


# Conversion to audio
from gtts import gTTS #Import Google Text to Speech
import os

  
# # The text that you want to convert to audio
# predicted_name = predict_names[0]
  
# # Language in which you want to convert
# language = 'en'
  
# # Passing the text and language to the engine, 
# # here we have marked slow=False. Which tells 
# # the module that the converted audio should 
# # have a high speed
# myobj = gTTS(text=predicted_name, lang=language, slow=False)
  
# # Saving the converted audio in a mp3 file named
# # name 
# myobj.save("D:/venv/name.mp3")
  
# # Playing the converted file
# os.system("D:/venv/name.mp3")

import pyttsx3
text_speech = pyttsx3.init()
predicted_name = predict_names[0]
text_speech.say(predicted_name)
text_speech.runAndWait()