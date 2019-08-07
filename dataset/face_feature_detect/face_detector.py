import cv2
import dlib
import numpy as np
from PIL import Image


PREDICTOR = './dataset/shape_predictor_68_face_landmarks.dat'
IMAGE = './dataset/bowen.JPG'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR)

img_cv2 = cv2.imread(IMAGE)

# number of faces in the image
rects = detector(img_cv2, 0)
print(len(rects))

# for the first face (rects[0]), get landmarks(p)
for p in predictor(img_cv2, rects[0]).parts():
    print(p.x, p.y)


for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_cv2, rects[i]).parts()])
    img_cv2 = img_cv2.copy()

    
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])

        cv2.circle(img_cv2, pos, radius=4, color=(0,255,0))

# .namedWindow("img_cv2")
cv2.imwrite('bowen_landmark.jpg', img_cv2)




