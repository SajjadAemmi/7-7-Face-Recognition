import cv2
import tensorflow as tf
import numpy as np
from mtcnn_face_detector import face_detect

model = tf.keras.models.load_model('saved_model')
names = ['Ali-Khamenei', 'Angelina-Jolie', 'Barak-Obama', 'Behnam-Bani',
         'Donald-Trump', 'Emma-Watson', 'Han-Hye-Jin', 'Kim-Jong-Un',
         'Leyla-Hatami', 'Lionel-Messi', 'Michelle-Obama', 'Morgan-Freeman',
         'Queen-Elizabeth', 'Scarlett-Johansson']

video = cv2.VideoCapture('input/kim.mp4')

while True:
    ret, frame = video.read()

    if ret == False:
        break

    x, y, a = face_detect(frame)

    if a != 0:
        img_face = frame[y:y + a, x:x + a].copy()
        cv2.rectangle(frame, (x, y), (x + a, y + a), (255, 0, 0), 4)

        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
        img_face = cv2.resize(img_face, (224, 224))
        img_face = img_face / 255.0
        img_face = img_face.reshape(1, 224, 224, 3)

        result = model.predict(img_face)
        index = np.argmax(result)
        print(names[index])

        cv2.putText(frame, names[index], (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    cv2.imshow('result', frame)

    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord('q') or ch == ord('Q'):
        break

cv2.waitKey()
