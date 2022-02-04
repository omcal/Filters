import cv2.cv2 as cv2
import numpy as np
import warnings

#to ignore the warning
warnings.filterwarnings("ignore")

np.seterr(over='ignore')

img = cv2.imread("lena.png")
cv2.imshow("Original", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


"""
In Kuwahara's algorithm I should use hsv color space. to calculate the mean of the pixels in the quadrant.
after finding the mean of the pixels in the quadrant, I should calculate the standard deviation of the pixels in the quadrant.
Then I find the minimum of the standard deviation 
Then I find the mean of the pixels in the quadrant which has the minimum standard deviation.
After that I convert pixel from hsv to bgr color space.
I calculate average of the pixels in the quadrant.
and I set the pixel to the average of the pixels in the quadrant.

"""

def average_color(color):
    color=cv2.cvtColor(color,cv2.COLOR_HSV2BGR)
    r,g,b =0,0,0
    for i in range(color.shape[0]):
        for j in range(color.shape[1]):
            r+=color[i,j,0]
            g+=color[i,j,1]
            b+=color[i,j,2]
    r=r//(color.shape[0]*color.shape[1])
    g=g//(color.shape[0]*color.shape[1])
    b=b//(color.shape[0]*color.shape[1])
    return (r,g,b)

def std(my_matrix):
    result=np.zeros(my_matrix.shape,dtype=np.float64)
    for i in range(my_matrix.shape[0]):
        for j in range(my_matrix.shape[1]):
            result[i,j]=my_matrix[i,j,2]
    return np.std(result)

def kuwara(my_matrix,window_size):
    r_offset = window_size // 2
    c_offset = window_size // 2
    # Border pixels

    img=cv2.copyMakeBorder(my_matrix,r_offset,r_offset,c_offset,c_offset,cv2.BORDER_REFLECT)
    # I create 4 quadrants with respect to the center of the middle pixel
    # also I include kernel size too
    for i in  range(my_matrix.shape[0]):
        for j in range(my_matrix.shape[1]):
            quand_a=img[i-r_offset:i,j-c_offset:j]
            quand_b=img[i-r_offset:i,j:j+c_offset]
            quand_c=img[i:i+r_offset,j-c_offset:j]
            quand_d=img[i:i+r_offset,j:j+c_offset]

            a_std=std(quand_a)
            b_std=std(quand_b)
            c_std=std(quand_c)
            d_std=std(quand_d)

            min_std=min(a_std,b_std,c_std,d_std)
            if min_std==a_std:
                my_matrix[i,j]=average_color(quand_a)
            if min_std==b_std:
                my_matrix[i,j]=average_color(quand_b)
            if min_std==c_std:
                my_matrix[i,j]=average_color(quand_c)
            if min_std==d_std:
                my_matrix[i,j]=average_color(quand_d)
    # its little bit tricky  I creted a border but after the loop I should remove it from the image
    # basically remove the border side pixels
    # and I added new border and its refect the image
    my_matrix=my_matrix[r_offset:-r_offset,c_offset:-c_offset]
    my_matrix=cv2.copyMakeBorder(my_matrix,r_offset,r_offset,c_offset,c_offset,cv2.BORDER_REFLECT)
    return my_matrix

img_out = kuwara(img,5)


#output of the image

img = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
cv2.imshow("Kuwahara", img_out)
cv2.waitKey(0)
