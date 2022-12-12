import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

# Reading the image with dlib
image_file = 'gd.jpeg'
img = cv2.imread(image_file)

# Face detection
faces = face_detector(img, 1)


# -------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
face_list = cascade.detectMultiScale(gray, minSize = (50,50))

boxes = []

for (x, y, w, h) in face_list:
    color = (0, 0, 225) 
    pen_w = 2 
    print(x, y, w, h)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
    
# -------

# Landmark detections

landmark_tuple = []
for k, d in enumerate(faces):
    landmarks = landmark_detector(img, d)
    for n in range(0, 27):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_tuple.append((x, y))
        cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

routes = []

for i in range(15, -1, -1):
    from_coordinate = landmark_tuple[i+1]
    to_coordinate = landmark_tuple[i]
    routes.append(from_coordinate)

from_coordinate = landmark_tuple[0]
to_coordinate = landmark_tuple[17]
routes.append(from_coordinate)

for i in range(17, 20):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)

from_coordinate = landmark_tuple[19]
to_coordinate = landmark_tuple[24]
routes.append(from_coordinate)

for i in range(24, 26):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)

from_coordinate = landmark_tuple[26]
to_coordinate = landmark_tuple[16]
routes.append(from_coordinate)
routes.append(to_coordinate)

# Route connecting

for i in range(0, len(routes)-1):
   from_coordinate = routes[i]
   to_coordinate = routes[i+1]
   img = cv2.line(img, from_coordinate, to_coordinate, (255, 255, 0), 1)

# Extract Facial area

mask = np.zeros((img.shape[0], img.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(np.bool_)

out = np.zeros_like(img)
out[mask] = img[mask]


# out --> 얼굴 형태 따놓은 상태
# out 좌표 움직여서 얼굴 변경

cv2.imshow('hi', out)
cv2.waitKey(0)