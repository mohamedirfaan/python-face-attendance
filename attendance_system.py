import numpy as np
import face_recognition
import cv2
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

import os
path = "./images"

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#    print("photos : ",classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

while(True):
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    cv2.imshow('frame', frame)
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            global name
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
 #           print(name)
            file=open("attendance.txt","r")
            content=list(file.read().split())
 #           print(content)
            file.close()
 #           print(name not in content)
            if name not in content:
                if(name=='irfaan' or name=="prasanna"):
                    file=open("attendance.txt","r")
                    content=list(file.read().split())
 #                   print(content)
                    file.close()
                    if name not in content:
                        file=open("attendance.txt","a")
                        from datetime import datetime
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        file.write(name+" "+str(current_time)+"\n")
                        file.flush()
                        file.close()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
