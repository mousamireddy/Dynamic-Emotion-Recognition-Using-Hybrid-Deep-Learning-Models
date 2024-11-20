#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array


# In[2]:


face_classifier=cv2.CascadeClassifier(r'C:\Users\Lenovo\Downloads\Project\haarcascade_frontalface_default.xml')
classifier=load_model('bidirectional.h5')


# In[3]:


emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
cap=cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
      print("Error: Video capture object is not opened.")
      exit()

    _, frame=cap.read()
    labels=[]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray_frame)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray=gray_frame[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=np.expand_dims(roi,axis=0)

            prediction=classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        else:
            cv2.putText(frame,'No Faces', (30,80),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




