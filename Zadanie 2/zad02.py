import numpy as np
import cv2 as cv
from ximea import xiapi
import glob


### CHESSBOARD ###
# termination criteria
def chessBoard():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpg')
    j =0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 5), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7, 5), corners2, ret)
            cv.imshow('img', img)
            cv.imwrite('sachovnica' + str(j) + '.png', img)
            j = j+1
            cv.waitKey(500)

    cv.destroyAllWindows()

    ret, mtx, dist,revx,tvx = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix: " + str(mtx))
    print("fx = " + str(mtx[0, 0]))
    print("fy = " + str(mtx[1, 1]))
    print("cx = " + str(mtx[0, 2]))
    print("cy = " + str(mtx[1, 2]))

    img = cv.imread('obrazok14.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('kalibrovany.png', dst)

#############################################################################

### CIRCLE DETECTION ###
def nothing(x):
    pass
def circleDetection():
    cam = xiapi.Camera()

    print('Opening first camera...')
    cam.open_device()

    #settings
    cam.set_exposure(10000)
    cam.set_param('imgdataformat','XI_RGB32')
    cam.set_param('auto_wb', 1)
    print('Exposure was set to %i us' %cam.get_exposure())

    #create instance of Image to store image data and metadata
    img = xiapi.Image()

    #start data acquisition
    print('Starting data acquisition...')
    cam.start_acquisition()

    cv.namedWindow('image')
    # create trackbars for color change
    cv.createTrackbar('dp', 'image', 1, 2, nothing)
    cv.createTrackbar('minDist', 'image', 70, 100, nothing)
    cv.createTrackbar('param1', 'image', 200, 300, nothing)
    cv.createTrackbar('param2', 'image', 40, 50, nothing)
    
    while True:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv.resize(image,(480,480))
        
        k = cv.waitKey(1)
        if k == ord('q'):
            break  
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Apply median blur to reduce noise
        gray_blur = cv.medianBlur(gray, 5)

        dp = cv.getTrackbarPos('dp','image')
        md = cv.getTrackbarPos('minDist','image')
        p1 = cv.getTrackbarPos('param1','image')
        p2 = cv.getTrackbarPos('param2','image')

        circles = cv.HoughCircles(gray_blur, cv.HOUGH_GRADIENT, dp, md, param1 = p1, param2 = p2, minRadius = 0, maxRadius = 0)
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
            # draw the outer circle
                cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # draw the center of the circle
                cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv.imshow('detected circles', image)

    cv.waitKey(0)
    cv.destroyAllWindows()

chessBoard()
circleDetection()