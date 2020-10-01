import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import glob
#import pyttsx3
from gtts import gTTS
import playsound

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
# I am not using this function as it was not very effective.
# But in certain circumstances, this can come handy
def speakAttendance(text, name,lang,slow=False):
    output = gTTS(text=text,lang=lang,slow=slow)
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename,block=False)

def markAttendance(name):
    with open('/Users/sanjeevviswanath/PycharmProjects/AttendanceSystem/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []

        ####Speech module
        #speakAttendance("Attendance Recorded for ",name,"en",False)
        #speaker = pyttsx3.init()
        #speaker.say("System has logged your attendance "+ name,name)
        #speaker.runAndWait()
        #speaker.stop()
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            speak("System has logged your attendance " + name)



path = "/Users/sanjeevviswanath/PycharmProjects/AttendanceSystem/FamilyImages/"

images = []
classNames = []

######

#imagePaths = [f for f in glob.glob('*.jpg')]
#print("Image path = ",imagePaths)
myList =[]
for fileName_relative in glob.glob(path+"**/*.j*",recursive=True):       ## first get full file name with directories using for loop
    print("Full file name with directories: ", fileName_relative)
    fileName_absolute = os.path.basename(fileName_relative)              ## Now get the file name with os.path.basename
    print("Only file name: ", fileName_absolute)
    myList.append(fileName_absolute)
#####


myList = os.listdir(path)
print("my list = ",myList)
print("os.path",os.path.splitext(myList[0]))
for cls in myList:
    print(f'{path}{cls}')
    curImg = cv2.imread(f'{path}{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print("ClassNames = ", classNames)


encodeListKnown = findEncodings(images)
print("Encode List for known images = ",encodeListKnown)
print("Encode List len = ",len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurrFrame = face_recognition.face_encodings(imgSmall,facesCurFrame)

    for encodeface, faceLoc in zip(encodesCurrFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeface)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y1-35), (x2, y2), (0, 255, 0),1)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)


#imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#faceLoc = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
#cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc)