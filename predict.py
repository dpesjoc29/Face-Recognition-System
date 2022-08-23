import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from random import choice
from numpy import load
from numpy import expand_dims
from face_embedding_4 import get_embedding
import pickle
from load_dataset_2 import extract_face
import json
from text_to_speech import text_to_speech
import sys

DATASET_DICT = './model_info/dataset_dect.json'
model_location = './models/model1.pkl'
# model_location = './models/actual_model.pkl'

def predict(input_img, node=False):
    load_face = extract_face(input_img)
    embedded_face = get_embedding(load_face)
    samples = expand_dims(embedded_face, axis=0)
    model = load_model()
    yhat_class = model.predict(samples)
    predicted_name = get_name(yhat_class)
    return predicted_name

def get_name(yhat_class):
    with open(DATASET_DICT, 'r') as f:
        dataset_dict = json.load(f)
    predict_class =  str(yhat_class[0])
    predict_name = dataset_dict[predict_class]
    # print(predict_class)
    return(predict_name)

def load_faces():
    data = load('D:/venv/dataset.npz')
    testX_faces = data['arr_2']
    return testX_faces

def load_model():
    with open(model_location,'rb') as f:
        models = pickle.load(f)
        return models


if __name__ == "__main__":
    # input_img = 'D:/venv/train/ben_afflek/httpcsvkmeuaeccjpg.jpg'
    # input_img = "D:/venv/train/jerry_seinfeld/httpgraphicsnytimescomimagessectionmoviesfilmographyWireImagejpg.jpg"
    input_img = sys.argv[1]
    # input_img = input('Enter image path: ')
    predicted_name = predict(input_img, True)
    # text_to_speech(predicted_name)
    print("dome")
    # text_to_speech(predicted_name)
