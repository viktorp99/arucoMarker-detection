import cv2
import cv2.aruco as aruco
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import glob
import os # file path handling

# select the camera to be processed
camId = 'alvium mono8'

# preconfugure some pathes 
basePath = './data/' + camId + '/'

# load camera paramerter to file 
# see https://docs.opencv.org/3.4/dd/d74/tutorial_file_input_output_with_xml_yml.html for details
fnCameraParams = basePath + 'calib_' + camId + '.yml'
s = cv2.FileStorage()
s.open(fnCameraParams, cv2.FileStorage_READ)
cameraMatrix = s.getNode('MTX').mat()
distCoeffs = s.getNode('DIST').mat()
s.release()
# length of the sides of a marker 
markerLength = 0.15 # [m]
axisLength = 0.1 # [m]





def findArucoMarkers(img, markerSize = 5, totalMarkers=250, draw=True):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ### Aruco
    ## dictionary 
    ## parameter
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    
    # detect marker and render
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    # draw marker detection 
    cv2.aruco.drawDetectedMarkers(img,bboxs,ids)
        
    # draw marker pose 
    # all metric dimensions are in meter
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(bboxs, markerLength, cameraMatrix, distCoeffs)
    if ids is not None:
        for (idx,rvec,tvec) in zip(ids,rvecs,tvecs):
                # TODO Aufgabe 3 - draw marker pose
                cv2.aruco.drawAxis(img,cameraMatrix,distCoeffs,rvec,tvec,axisLength)
        
    return [bboxs, rvecs, tvecs, ids] 
        



def arucoAugBox(rvecs,tvecs,imgBGR):
    axis = np.float32([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0], [0.1, -0.1, 0],
                   [-0.1, -0.1, 0.1], [-0.1, 0.1, 0.1], [0.1, 0.1, 0.1],[0.1, -0.1, 0.1]])
    # Now we transform the cube to the marker position and project the resulting points into 2d
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    
    # Now comes the drawing. 
    # In this example, I would like to draw the cube so that the walls also get a painted
    # First create six copies of the original picture (for each side of the cube one)
    side1 = imgBGR.copy()
    side2 = imgBGR.copy()
    side3 = imgBGR.copy()
    side4 = imgBGR.copy()
    side5 = imgBGR.copy()
    side6 = imgBGR.copy()
    
    # Draw the bottom side (over the marker)
    side1 = cv2.drawContours(side1, [imgpts[:4]], -1, (255, 0, 0), -2)
    # Draw the top side (opposite of the marker)
    side2 = cv2.drawContours(side2, [imgpts[4:]], -1, (255, 0, 0), -2)
    # Draw the right side vertical to the marker
    side3 = cv2.drawContours(side3, [np.array(
        [imgpts[0], imgpts[1], imgpts[5],
         imgpts[4]])], -1, (255, 0, 0), -2)
    # Draw the left side vertical to the marker
    side4 = cv2.drawContours(side4, [np.array(
        [imgpts[2], imgpts[3], imgpts[7],
         imgpts[6]])], -1, (255, 0, 0), -2)
    # Draw the front side vertical to the marker
    side5 = cv2.drawContours(side5, [np.array(
        [imgpts[1], imgpts[2], imgpts[6],
         imgpts[5]])], -1, (255, 0, 0), -2)
    # Draw the back side vertical to the marker
    side6 = cv2.drawContours(side6, [np.array(
        [imgpts[0], imgpts[3], imgpts[7],
         imgpts[4]])], -1, (255, 0, 0), -2)
    
    imgBGR = cv2.drawContours(imgBGR, [imgpts[:4]], -1, (255, 255, 0), -2)
    for i, j in zip(range(4), range(4, 8)):
        imgBGR = cv2.line(imgBGR, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 2)
        imgBGR = cv2.drawContours(imgBGR, [imgpts[4:]], -1, (255, 255, 0), 2)
        
    # Until here the walls of the cube are drawn in and can be merged
    imgBGR = cv2.addWeighted(side1, 0.1, imgBGR, 0.9, 0)
    imgBGR = cv2.addWeighted(side2, 0.1, imgBGR, 0.9, 0)
    imgBGR = cv2.addWeighted(side3, 0.1, imgBGR, 0.9, 0)
    imgBGR = cv2.addWeighted(side4, 0.1, imgBGR, 0.9, 0)
    imgBGR = cv2.addWeighted(side5, 0.1, imgBGR, 0.9, 0)
    imgBGR = cv2.addWeighted(side6, 0.1, imgBGR, 0.9, 0)
    
    return imgBGR

cap = cv2.VideoCapture(0)
#imgAug = cv2.imread(r"C:\Users\Viktor\Pictures\001_-_Mystery_Box-512.png")
while True:
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    # loop through all the markers and augment each one
    if len(arucofound[0])!=0:
       for rvec,tvec, id in zip(arucofound[1], arucofound[2], arucofound[3]):
            img = arucoAugBox(rvec, tvec, img)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


def arucoAug(bbox, id, img, imgAug, drawId = True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgout = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgout = img + imgout
    return imgout