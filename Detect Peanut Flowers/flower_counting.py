
import cv2
import numpy as np
import os
import sys
import argparse as ap


def kMeans(image):

      # Pre processing image
      #med_img = cv2.medianBlur(image , 5)
      gauss_img = cv2.GaussianBlur(image , (7 , 7) , 0)
      k = 7
      vector_img = gauss_img.reshape(-1 , 3)

      # Cast as float for K means (Mean computation)
      vector_img = np.float32(vector_img)

      # OpenCV specific params
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 0 , 1.0)

      ret , label , clusters = cv2.kmeans(vector_img , k , None , criteria,
                                       15 , cv2.KMEANS_RANDOM_CENTERS)

      #print(label.shape)
      res = clusters[label.flatten()]

      clust_img = res.reshape((image.shape))

      
      cv2.imshow('clus_img' , clust_img.astype(np.uint8))
      cv2.waitKey(0)
      cv2.destroyAllWindows()     

      return label.reshape((image.shape[0],image.shape[1])) , clust_img.astype(np.uint8)


def bgr_hsv_convert(b , g , r):

      color = np.uint8([[[b , g , r]]])
      hsv_color = cv2.cvtColor(color , cv2.COLOR_BGR2HSV)

      hue = hsv_color[0][0][0]

      lower = [hue - 10  , 100 , 100]
      upper = [hue + 10 , 255 , 255]

      return lower , upper


def colorDetect(image , low_range , high_range):

      mask = cv2.inRange(image , low_range , high_range)
      output = cv2.bitwise_and(image , image , mask = mask)

      cimg = cv2.cvtColor(output.copy(),cv2.COLOR_BGR2GRAY)
      img = cv2.medianBlur(cimg,5)

      #print(mask)
      #print(output)

      #print(np.array_equal(mask , output))
      #print(cv2.countNonZero(mask))

      print('max color: ' , np.amax(output))

      

      '''circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

      circles = np.uint16(np.around(circles))
      for i in circles[0,:]:
          # draw the outer circle
          cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
          # draw the center of the circle
          cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

      cv2.imshow('detected circles',img)'''
      
            

      #kernelOpen=np.ones((5,5))
      #kernelClose=np.ones((20,20))

      #gray_img = cv2.cvtColor(output.copy(), cv2.COLOR_BGR2GRAY)

      #kernel = np.ones((2 , 2) , np.uint8)
      #erosion = cv2.erode(np.uint8(output.copy()) , kernel , iterations = 7)
      #retVal , threshold_img = cv2.threshold(gray_img , np.average(gray_img) , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      #median_filter_img = cv2.medianBlur(threshold_img , 9)      

      #maskOpen=cv2.morphologyEx(output,cv2.MORPH_OPEN,kernelOpen)
      #maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)      

      #cv2.imshow('mask' , mask)
      cv2.imshow('out' , output)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      if np.amax(output) > 242:
            return output

      return None
      
def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

if __name__ == '__main__':

      #print(os.listdir())

      parser = ap.ArgumentParser()
      parser.add_argument('-i' , '--input' , required = True)

      args = parser.parse_args()
     
      

      #img_path = [f for f in os.listdir() if f.endswith('.JPG')]

      #print(img_path)

      for im in [args.input]:#]img_path:

            img = cv2.imread(im)
            img = cv2.resize(img , (800 , 640))

            print(im , img.shape)

            #img_lab = cv2.cvtColor(img , cv2.COLOR_BGR2LAB)
            img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

            #print(img_lab)
            
            '''cv2.imshow('img' , img_lab)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            

            #label , k_img = kMeans(img_lab)
            low_range , high_range = bgr_hsv_convert(106 , 241 , 255)

            # Yellow color min and max range
            # min bgr [0 , 171 , 206]
            # max bgr [115, 241, 255]
            #low_range = np.array([181 , 128 , 202] , dtype = np.uint8)
            #high_range = np.array([240 , 117 , 193] , dtype = np.uint8)

            low_range = np.array(low_range , dtype = np.uint8)
            high_range = np.array(high_range , dtype = np.uint8)

            print(low_range , high_range)
            output = None

            output = colorDetect(img_hsv , low_range , high_range)

            label = None
            k_img = None



            if output is not None:
                  
                  label , k_img = kMeans(output)

                  #print(label)                  

                  for i in range(2):
                 
                       arr = np.zeros(img.shape , np.uint8)
                       arr[label == i] = img[label == i]
                       #print(arr , np.amax(arr) , np.average(arr) , i)
                       cv2.imshow(str(i) , arr)
                       cv2.waitKey(0)          


                  arr = np.zeros(img.shape , np.uint8)
                  arr[label == 1] = img[label == 1]
                  #cv2.imshow('flower' , arr)
                  #cv2.waitKey(0)

                  gray_img = cv2.cvtColor(arr , cv2.COLOR_BGR2GRAY)

                  #cv2.imshow('gray' , gray_img)
                  #cv2.waitKey(0)
      
                  retVal , threshold_img = cv2.threshold(gray_img , np.average(gray_img) ,
                                             255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)


                  #cv2.imshow('thresh' , threshold_img)
                  #cv2.waitKey(0)

                  median_filter_img = cv2.medianBlur(threshold_img , 3)

                  #cv2.imshow('med' , median_filter_img)
                  #cv2.waitKey(0)

                  tmp , cnt , hier = cv2.findContours(median_filter_img, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
            

                  sorted_cnts = sorted(cnt , key = cv2.contourArea , reverse = True)

                  print(len(sorted_cnts))

                  for c in sorted_cnts:
                        print('peri: ' , cv2.arcLength(c , True))
                        approx = cv2.approxPolyDP(c , 0.01 * cv2.arcLength(c , True), True)
                        print('contour len: ' , len(approx))
                        cv2.drawContours(img, [c], -1, (255, 0, 0), 3)
                        #cv2.imshow(str(i) , 

            
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  result_str = 'Flower Count: ' + str(len(sorted_cnts))
                  cv2.putText(img , result_str ,( 10 , img.shape[0] - 10) , font , 1 ,(0 , 255 , 255) , 3 , cv2.LINE_AA)
                  cv2.imshow('cnt' , img)

                  file_nm = im[0 : im.index('.')] + '_count.png'
                  print(file_nm , len(sorted_cnts))
                  cv2.imwrite(file_nm , img)
            
                  #sorted_cnts = []
      
                  #ret , thresh = cv2.threshold()
            else:
                  print('No flowers detected')

 
