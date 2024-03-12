from __future__ import print_function
import sys
import time
import cv2
import numpy as np

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value

def sharpen(my_image):
    my_image = cv2.cvtColor(my_image, cv2.CV_8U)
    height, width, n_channels = my_image.shape

    height = height // 2
    width = width // 2
    result = np.zeros(my_image.shape, my_image.dtype)

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            for k in range(0, n_channels):
                    sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k]  \
                                - my_image[j - 1, i, k] - my_image[j, i + 1, k]\
                                - my_image[j, i - 1, k]
                    result[j, i, k] = saturated(sum_value)

    result[height:, :, :] = my_image[height:, :, :]
    result[:, width:, :] = my_image[:, width:, :]
 
    return result


def rotate(my_image):

    my_image = cv2.cvtColor(my_image, cv2.CV_8U)
    height, width, n_channels = my_image.shape

    height = height // 2
    width = width // 2
    result = np.zeros(my_image.shape, my_image.dtype)

    for j in range(1, height - 1):
        for i in range(width-1, (width-1)*2):
            for k in range(0, n_channels):
                result[j, i, k] = my_image[i, j, k]

    result[height:, :, :] = my_image[height:, :, :]
    result[:, :width, :] = my_image[:, :width, :]
    
    return result

# Create VideoCapture object to capture video from the default camera (index 0)
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

j = 1
end = 0

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Display the captured frame
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if (k == ord(" ")  and j < 5):
        cv2.imwrite("obrazok" + str(j) + ".jpg", frame)
        j += 1

    if k == ord('q'):
        break  # Exit the loop if 'q' is pressed
    
    if j == 5 and end == 0:
        image1 = cv2.imread("obrazok1.jpg")
        image2 = cv2.imread("obrazok2.jpg")
        image3 = cv2.imread("obrazok3.jpg")
        image4 = cv2.imread("obrazok4.jpg")

        height, width = 240, 240
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))
        image3 = cv2.resize(image3, (width, height))
        image4 = cv2.resize(image4, (width, height))

        combined_image = np.zeros((2*240, 2*240, 3), dtype=np.uint8)
        combined_image[:240, :240] = image1
        combined_image[:240, 240:] = image2
        combined_image[240:, :240] = image3
        combined_image[240:, 240:] = image4
     
        cv2.imwrite("mozaika.jpg", combined_image)
        src = cv2.imread("mozaika.jpg")

        blue, green, red = cv2.split(src[240:, 240:])
        src[240:, 240:] = cv2.merge([ green * 0, blue * 0,red])

        dst0 = sharpen(src)
        dst0 = rotate(dst0)
        
        cv2.imshow("Filter", dst0)
    
        cv2.imshow("MozaikaOriginals", combined_image)
        end = 1

        # Výpis dátového typu, rozmerov a veľkosti obrazu
        print("Dátový typ obrazu:", dst0.dtype)
        print("Rozmery obrazu:", dst0.shape)
        print("Veľkosť obrazu (v pixeloch):", dst0.size)

# Release the camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()