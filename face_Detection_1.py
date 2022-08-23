import cv2
from mtcnn.mtcnn import MTCNN
from predict import predict
from text_to_speech import text_to_speech
detector = MTCNN()
import time



if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    # to change resolution
    # def make_480p():
    #     video.set(3,100)
    #     video.set(4,200)
   
    # make_480p()
    # to access webcam from phone
    # address = 'https://192.168.1.17:8080/video'
    # video.open(address)
    if (video.isOpened() == False):
        print("Web Camera not detected")
    while (True):
        ret, frame = video.read()
        cv2.imwrite("imageframe.jpg",frame)
        frame = cv2.flip(frame, 1)
        # print(type(frame))
        if ret == True:
            location = detector.detect_faces(frame)
            # print(location)
            if len(location) > 0:
                for face in location:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    x, y, width, height = face['box']
                    x2, y2 = x + width, y + height
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    t1 = time.time()
                    try:
                        predicted_name =  predict('./imageframe.jpg')
                        cv2.putText(frame, 
                            predicted_name, 
                            (x, y), 
                            font, 0.5, 
                            (0, 128, 0), 
                            2, 
                            cv2.LINE_4
                        )
                        text_to_speech(predicted_name)
                        t2 = time.time()
                        print(t2 - t1)
                    except:
                        pass
            cv2.imshow("Output",frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            key = cv2.waitKey(20)
            if key == ord('q'):
                break
        # else:
        #     break
    video.release()
    cv2.destroyAllWindows()