from face_embedding_4 import save_embedding
from load_dataset_2 import save_dataset
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

MODEL_PATH = './models/model1.pkl'
TRAIN_PATH = './dataset/train/'
TEST_PATH = './dataset/test/'
EMBEDDING_PATH = './model_info/faces_embeddings.npz'
DATASET_PATH =	'./model_info/dataset.npz'
DICT_PATH = './model_info/dataset_dect.json'

def train(dataset_path, embedding_path, model_path, train_path, test_path, dict_path):
    # save dataset
    save_dataset(train_path=train_path, test_path=test_path, save_path=dataset_path, dict_path=dict_path)
    # # save embedding
    save_embedding(save_path=embedding_path, dataset_path=dataset_path)
    # load face embeddings
    data = load(embedding_path)
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors 
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    # testy = out_encoder.transform(testy)
    # fit model
    # print(trainy)
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    with open(model_path,'wb') as f:
        pickle.dump(model,f)
    print('Training completely sucessfully')
    
if __name__ == "__main__":
    train(DATASET_PATH, EMBEDDING_PATH, MODEL_PATH, TRAIN_PATH, TEST_PATH, DICT_PATH) 

