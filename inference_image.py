import cv2
import tensorflow as tf
import numpy as np
from mtcnn_face_detector import face_detect
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input image path', default='input/obamas.jpg', type=str)
parser.add_argument('--output', help='output image path', default='output/result.jpg', type=str)
args = parser.parse_args()

model = tf.keras.models.load_model('saved_model')
names = ['Ali-Khamenei', 'Angelina-Jolie', 'Barak-Obama', 'Behnam-Bani',
         'Donald-Trump', 'Emma-Watson', 'Han-Hye-Jin', 'Kim-Jong-Un',
         'Leyla-Hatami', 'Lionel-Messi', 'Michelle-Obama', 'Morgan-Freeman',
         'Queen-Elizabeth', 'Scarlett-Johansson']

frame = cv2.imread(args.input)

resuls = face_detect(frame)
for (x, y, a) in resuls:
    img_face = frame[y:y + a, x:x + a].copy()
    cv2.rectangle(frame, (x, y), (x + a, y + a), (255, 0, 0), 4)

    img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
    img_face = cv2.resize(img_face, (224, 224))
    img_face = img_face / 255.0
    img_face = img_face.reshape(1, 224, 224, 3)

    result = model.predict(img_face)
    index = np.argmax(result)
    print(names[index])

    cv2.putText(frame, names[index], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

cv2.imwrite(args.output, frame)
cv2.imshow('result', frame)
cv2.waitKey()
