import cv2

cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
face_list = cascade.detectMultiScale(gray, minSize = (50,50))

for (x, y, w, h) in face_list:
    color = (0, 0, 225) 
    pen_w = 2 
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
    
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()