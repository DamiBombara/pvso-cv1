import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def checkImage(img):
    if img is None:
        print("Could not find the image!")
        return -1


def openCVfunction(argv):
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
 
    imageName = argv
    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image

    checkImage(src)

    src = cv.GaussianBlur(src, (3, 3), 0)

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)

    abs_dst = cv.convertScaleAbs(dst)

    cv.imshow(window_name, abs_dst)
    cv.imwrite("funkcia.png", abs_dst)
    cv.waitKey(0)


def gaussian_blur(img):
    height, width = img.shape[:2]
    gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = np.zeros((height, width), np.uint8)

    for i in range(2, height - 2):
        for j in range(2, width - 2):
            sum = (2 * gr[i - 2, j - 2] +  4 *  gr[i - 2, j - 1] +  5 *  gr[i - 2, j] +  4 * gr[i - 2, j + 1] + 2 * gr[i - 2, j + 2] + 
                   4 * gr[i - 1, j - 2] +  9 *  gr[i - 1, j - 1] + 12 *  gr[i - 1, j] +  9 * gr[i - 1, j + 1] + 4 * gr[i - 1, j + 2] + 
                   5 * gr[i    , j - 2] + 12 *  gr[i    , j - 1] + 15 *  gr[i    , j] + 12 * gr[i    , j + 1] + 5 * gr[i    , j + 2] + 
                   4 * gr[i + 1, j - 2] +  9 *  gr[i + 1, j - 1] + 12 *  gr[i + 1, j] +  9 * gr[i + 1, j + 1] + 4 * gr[i + 1, j + 2] + 
                   2 * gr[i + 2, j - 2] +  4 *  gr[i + 2, j - 1] +  5 *  gr[i + 2, j] +  4 * gr[i + 2, j + 1] + 2 * gr[i + 2, j + 2]) // 159
            
            blur[i, j] = np.clip(sum, 0, 255)

    return blur


def LoG(image):
    img = cv.imread(image)
    checkImage(img)
    img = gaussian_blur(img)

    height, width = img.shape[:2]
    laplaceOfGaussian = np.zeros((height, width), np.uint8)

    for i in range(2, height - 2):
        for j in range(2, width - 2):
            sum = (-2 * img[i     , j + 1] - 2 * img[i    , j - 1] - 2 *  img[i + 1, j    ] - 2 * img[i - 1, j    ] - img[i     , j + 2] - 
                        img[i     , j - 2] -     img[i + 2, j    ] -      img[i - 2, j    ] -     img[i - 1, j + 1] - img[i + 1 , j + 1] - 
                        img[i + 1 , j - 1] -     img[i - 1, j - 1] + 16 * img[i    , j    ])
            
            laplaceOfGaussian[i, j] = np.clip(sum, 0, 255)
    
    cv.imshow("laplacian of gaussian", laplaceOfGaussian)
    cv.imwrite("laplacian_of_gaussian.png", laplaceOfGaussian)
    cv.waitKey(0)


def show_rgb_histogram(image):
    img = cv.imread(image)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    plt.figure(figsize = (10, 5))

    plt.subplot(1, 3, 1)
    plt.hist(r_channel.ravel(), color = 'red', bins = 256, alpha = 0.5)
    plt.xlabel('Intenzita')
    plt.ylabel('Počet pixelov')
    plt.title('Histogram kanálu R')

    plt.subplot(1, 3, 2)
    plt.hist(g_channel.ravel(), color = 'green', bins = 256, alpha = 0.5)
    plt.xlabel('Intenzita')
    plt.ylabel('Počet pixelov')
    plt.title('Histogram kanálu G')

    plt.subplot(1, 3, 3)
    plt.hist(b_channel.ravel(), color = 'blue', bins = 256, alpha = 0.5)
    plt.xlabel('Intenzita')
    plt.ylabel('Počet pixelov')
    plt.title('Histogram kanálu B')

    plt.tight_layout()
    plt.show()

#openCVfunction("dog.png")
#LoG("dog.png")
#show_rgb_histogram("dog.png")