import cv2.cv2 as cv2
import numpy as np
np.seterr(over='ignore')

img = cv2.imread("lena.png", cv2.IMREAD_COLOR)

cv2.imshow("Original", img)
# Create a Gaussian kernel with kernel size and sigma which is passed as arguments
# The Gaussian kernel is a 2D array and is returned as an nxn array

def Gauissian_kernel(sigma, size):
    # I used mgrid  to create a 2D array also linspace can be used too
    # it return two arrays and  we called meshgrid
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # calculate the gaussian kernel
    # dynamic sigma value at above its std of the gaussian kernel

    #if I need to use the std of kernel

    #sigma=np.std(x**2+y**2)

    gauss_kernel=  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * (1 / (2 * np.pi * sigma**2))
   # print(sigma)
    return gauss_kernel

# its hearth of gaussian filter
# its a self written filter2d I called PoorFilter2D because it is not a built in function nor a opencv function
# Its slow but it works If I could use fourier transform it would be faster
def PoorFilter2D(img, kernel):
    r=img.shape[0]
    c=img.shape[1]
    # I create two offset to shift the kernel
    r_offset = kernel.shape[0]//2
    c_offset=kernel.shape[1]//2
    # I create a new image with the same size as the original image to store the filtered image
    img_out = np.zeros((r,c,3), dtype=np.uint8)
    # I added border to the image to solve edge sitation

    img=cv2.copyMakeBorder(img,r_offset,r_offset,c_offset,c_offset,cv2.BORDER_REPLICATE)
    # I loop through the image and apply the filter
    # sum is used to sum the kernel and multiply is used to multiply the kernel with the image pixel and I assigned the result to the new image
    for i in range(r_offset,r+r_offset):
        for j in range(c_offset,c+c_offset):
            sum=0
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    sum+=kernel[k,l]*img[i-r_offset+k,j-c_offset+l]
            img_out[i-r_offset,j-c_offset] = (sum)
    return img_out


# I create a function with using gaussian kernel and PoorFilter2D  its return a image with gaussian filter
def Gauissian_filter(img, sigma, size):
    kernel = Gauissian_kernel(sigma, size)
    #  I divide the kernel by the sum of the kernel to calculate the gauissian filter
    kernel = kernel / np.sum(kernel)
    return PoorFilter2D(img, kernel)

# output of gaussian filter
img_blur = Gauissian_filter(img, sigma=2, size=3)
cv2.imshow("Gaussian",img_blur)
cv2.waitKey(0)
