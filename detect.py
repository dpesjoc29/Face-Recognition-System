import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
import sys

if __name__ == "__main__":
    file = sys.argv[1]
    frame = cv2.imread(file)
    location = detector.detect_faces(frame)
    print(location)