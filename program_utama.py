import cv2
import face_recognition

imgAsal = face_recognition.load_image_file('foto/mark1.jpeg')
imgAsal = cv2.cvtColor(imgAsal, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('foto_tes/elon2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgAsal)[0]
encodeAsal = face_recognition.face_encodings(imgAsal)[0]
cv2.rectangle(imgAsal, (faceLocation[3], faceLocation[0]),
              (faceLocation[1], faceLocation[2]), (255, 0, 255), 2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]),
              (faceLocationTest[1], faceLocationTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeAsal], encodeTest)
faceDis = face_recognition.face_distance([encodeAsal], encodeTest)

print('Mirip :', results[0])

imS1 = cv2.resize(imgAsal, (540, 540))
imS2 = cv2.resize(imgTest, (540, 540))

cv2.imshow('Image Asal', imS1)
cv2.imshow('Image Test', imS2)
cv2.waitKey(0)
