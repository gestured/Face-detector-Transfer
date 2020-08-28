# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import cv2
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('-i', '--frame', required=True, help= 'Input the image path, if using video stream enter "None"')
parser.add_argument('-c', '--confidence', default=0.5, type= float)

args= vars(parser.parse_args())

print('Loading Model')
model= cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
if args['frame']== 'None':
    
    cam= cv2.VideoCapture(0)
    while(True):
        grab, frame= cam.read()
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        print('Computing Detections')
        (h,w)= frame.shape[:2]
        model.setInput(blob)
        predictions= model.forward()
        
        for i in range(0, predictions.shape[2]):
            confidence= predictions[0][0][i][2]
            if confidence > args['confidence']:
                
                box= predictions[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY)= box.astype('int')
                
                text= '{:.2f}%'.format(confidence*100)
                y= startY-10 if startY-10>10 else startY+10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255),2)
                cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1)== ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
else:
    
    frame= cv2.imread(args['frame'])
    (h,w)= frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
    print('Computing Detections')
    model.setInput(blob)
    predictions= model.forward()
        
    for i in range(0, predictions[2]):
        confidence= predictions[0][0][i][2]
        if confidence > args['confidence']:
            box= predictions[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY)= box.astype('int')
            
            text= '{:.2f}%'.format(confidence*100)
            y= startY-10 if startY-10>10 else startY+10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255),2)
            cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow('Output', frame)
    
                
        
        
        
    
    


