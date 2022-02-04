import cv2.cv2 as cv2
import numpy as np
np.seterr(over='ignore')

img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
cv2.imshow("Original", img)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# its a basic mean filter
# I loop through the image and calculate the mean of the pixel values
# to avoid the border values I start at kernel_size/2 and end at the image size - kernel_size/2
# then I calculate the mean of the pixel values in the kernel

def meanFilter(img, kernel):
    kernel_size = kernel // 2
    for i in range(kernel_size, img.shape[0] - kernel_size):
        for j in range(kernel_size, img.shape[1] - kernel_size):
            for k in range(img.shape[2]):
                matrix=np.zeros((kernel,kernel))
                for x in range(kernel):
                    for y in range(kernel):
                        matrix[x][y]=img[i-kernel_size+x][j-kernel_size+y][k]
                img[i][j][k]=np.mean(matrix)

img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

meanFilter(img, 9)
# Result of mean filter
cv2.imshow("mean", img)
cv2.waitKey(0)
