# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
MODEL = load_model('./dataset/facenet_keras.h5')
SAVE_EMBEDDING_PATH = './model_info/faces_embeddings.npz'
DATASET_PATH =	'./model_info/dataset.npz'
# get the face embedding for one face
def get_embedding(face_pixels, model=MODEL):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def save_embedding(save_path, dataset_path):
	# load the face dataset
	# data = load('D:/venv/dataset.npz')
	data = load(dataset_path)
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
	# load the facenet model
	model = MODEL
	print('Loaded Model')
	# convert each face in the train set to an embedding
	newTrainX = [] 
	for face_pixels in trainX:
		embedding = get_embedding(face_pixels, model)
		newTrainX.append(embedding)
	# convert each face in the test set to an embedding
	newTrainX = asarray(newTrainX)
	print(newTrainX.shape)
	newTestX = list()
	for face_pixels in testX:
		embedding = get_embedding( face_pixels,model)
		newTestX.append(embedding)
	newTestX = asarray(newTestX)
	print(newTestX.shape)
	# save arrays to one file in compressed format
	savez_compressed(save_path, newTrainX, trainy, newTestX, testy)

if __name__ == '__main__':
	save_embedding(SAVE_EMBEDDING_PATH, DATASET_PATH)