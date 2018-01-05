# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft.helper import fftshift

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")  # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here
   img_out = img_in
   row, col, channel = img_in.shape
   cdf = []
   for i in range(0, channel):
    img_temp = np.empty((row, col))
    histr = cv2.calcHist([img_in], [i], None, [256], [0, 256])
    cumulative_sum = np.cumsum(histr)
    cv2.normalize(cumulative_sum, cumulative_sum, 0, 255, cv2.NORM_MINMAX)
    cdf_normalized = np.int32(np.around(cumulative_sum))
    for k in range(0, row):
        for j in range(0, col):
            val = img_in[k, j][i]
            img_temp[k, j] = cdf_normalized[val]
    cdf.append(img_temp)
   img_out = cv2.merge((cdf[0], cdf[1], cdf[2]))
   return True, img_out

def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return True, cv2.merge((blue, green, red))

def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # succeed, output_image = histogram_equalize(input_image)
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def low_pass_filter(img_in):
    
   # Write low pass filter here
   row, col, channel = img_in.shape
   img_in_gray = img_in
   if channel > 2:
       img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   fft_shift = ft(img_in_gray)
   magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
   
   rows, cols = img_in_gray.shape
   crows, ccols = rows / 2, cols / 2
   zeros = np.zeros((rows, cols))
   zeros[crows - 10:crows + 10, ccols - 10:ccols + 10] = 1
   ift_shift = fft_shift * zeros
   img_out = ift(ift_shift)
   
   """plt.subplot(131), plt.imshow(img_in_gray, cmap='gray')
   plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
   plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   plt.subplot(133), plt.imshow(img_out, cmap='gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   
   plt.show()"""
   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   row, col, channel = img_in.shape
   img_in_gray = img_in
   if channel > 2:
       img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)   
   fft_shift = ft(img_in_gray)
   magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
   
   rows, cols = img_in_gray.shape
   crows, ccols = rows / 2, cols / 2
   fft_shift[crows - 10:crows + 10, ccols - 10:ccols + 10] = 0

   img_out = ift(fft_shift)
   
   """plt.subplot(131), plt.imshow(img_in_gray, cmap='gray')
   plt.title('Gray Image'), plt.xticks([]), plt.yticks([])
   plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   plt.subplot(133), plt.imshow(img_out, cmap='gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   
   plt.show()"""
   return True, img_out
   
def deconvolution(img_in):
   
   # Write deconvolution codes here
   gk = cv2.getGaussianKernel(21, 5)
   gk = gk * gk.T
   rows, cols = img_in.shape
   imf = ft(img_in, (rows, cols))  # make sure sizes match
   gkf = ft(gk, (rows, cols))  # so we can multiple easily
   magnitude_spectrum = 20 * np.log(np.abs(gkf))
   imconvf = imf/gkf
   
   img_out = ift(imconvf)
   img_out = img_out*255
   """plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   plt.subplot(122), plt.imshow(img_out, cmap='gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   plt.show()"""
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   img_out = img_in1  # Blending result
   A = img_in1[:, :img_in1.shape[0]]
   B = img_in2[:img_in1.shape[0], :img_in1.shape[0]]
   
   # generate Gaussian pyramid for A
   G = A.copy()
   gpA = [G]
   for i in xrange(6):
       G = cv2.pyrDown(G)
       gpA.append(G)
       
   # generate Gaussian pyramid for B 
   G = B.copy()
   gpB = [G]
   for i in xrange(6):
       G = cv2.pyrDown(G)
       gpB.append(G)
    
   # generate Laplacian Pyramid for A
   lpA = [gpA[5]]
   for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    
    # generate Laplacian Pyramid for B
   lpB = [gpB[5]]
   for i in xrange(5, 0, -1):
       GE = cv2.pyrUp(gpB[i])
       L = cv2.subtract(gpB[i - 1], GE)
       lpB.append(L)
        
   # Now add left and right halves of images in each level
   LS = []
   for la, lb in zip(lpA, lpB):
       rows, cols, dpt = la.shape
       ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
       LS.append(ls)
   # now reconstruct
   ls_ = LS[0]
   for i in xrange(1, 6):
       ls_ = cv2.pyrUp(ls_)
       ls_ = cv2.add(ls_, LS[i])
   # image with direct connecting each half
   real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))
   return True, ls_

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
      
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
         sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
          help_message()
          sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
          help_message()
          sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
          print("Input parameters out of bound ...")
          sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
