# Detect edges

import cv2
import numpy as np
import scipy.io
import scipy.misc
import os
import tifffile as tiff
from PIL import Image
#from PythonMagick import Image as pyImage
#import png


#tif_files = [f for f in os.listdir() if f.endswith('.tif')]
mat_files = [m for m in os.listdir() if m.endswith('.MAT')]
#a = sorted([str(x) for x in range(50 , 150 , 10)])

#print(tif_files)

avg_img = []

test_img = np.zeros((512 , 424) , dtype = np.uint16)

avg_mat = None

counter = 0

# Calculate the average of the images
for  f in mat_files:
      

      print(f)

      #print(t)
      #mat_arr = (scipy.io.loadmat(m)[a[counter]])
    
      mat_arr = (scipy.io.loadmat(f)['depthmat'])

      print(scipy.io.loadmat(f))
      print(mat_arr , np.amax(mat_arr))

      #print(scipy.io.loadmat(m))

      depth_img = mat_arr
      img = scipy.misc.toimage(mat_arr, high=np.max(mat_arr), low=np.min(mat_arr), mode='I')
      img.save('my16bit.png')

      test_img = mat_arr

      #print(img , img.dtype)

      #cv2.imshow('img' , mat_arr)
      

      cv2.imwrite(f[0 : f.index('.')] + '.png' , mat_arr)
      #pyImage('test_depth_mat.png').write("foo.png")
      cv2.imwrite(f[0 : f.index('.')] + '.tiff' , mat_arr)
      #png.from_array(mat_arr, mode='L;16').save('foo.png')

      counter += 1

      kernel = np.ones((9 , 9) , np.uint8)
      #blur_img = cv2.cv2.bilateralFilter(np.uint8(depth_img.copy()), 9 , 225 , 225)
      #erosion = cv2.erode(blur_img , kernel , iterations = 12)
      #erosion_new = cv2.erode(erosion , kernel , iterations = 12)
      erosion = cv2.erode(np.uint8(depth_img.copy()) , kernel , iterations = 8)

      retVal , threshold_img = cv2.threshold(erosion , np.average(np.uint8(erosion.copy())) , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      #retVal , threshold_img = cv2.threshold(np.uint8(depth_img.copy()) ,
      #                                       np.average(np.array(depth_img.copy() , dtype = np.uint8)) ,
       #                                      255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      median_filter_img = cv2.medianBlur(threshold_img , 59)
      #erosion_new = cv2.erode(median_filter_img , kernel , iterations = 16)

      #img , cnt , hier = cv2.findContours(median_filter_img, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
      img , cnt , hier = cv2.findContours(median_filter_img, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

      sorted_cnts = sorted(cnt , key = cv2.contourArea , reverse = True)

      #closing = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
      
      #cv2.imshow('ero_8' , np.uint8(erosion.copy()))

      #cv2.imshow('filtered' , median_filter_img)

      #cv2.imshow('closing' , closing)

      #print(len(sorted_cnts))

      approx_max = {}
      cnts_list = []
      #ero_img = cv2.cvtColor(np.uint8(median_filter_img.copy()),cv2.COLOR_GRAY2RGB)
      ero_img = cv2.cvtColor(np.uint8(erosion.copy()),cv2.COLOR_GRAY2RGB)
      for c in sorted_cnts:

            approx = cv2.approxPolyDP(c , 0.02 * cv2.arcLength(c , True) , True)
            approx_max[len(approx)] = c
            cnts_list.append(len(approx))


      cv2.waitKey(0)
      cv2.destroyAllWindows()
      #print(cnts_list)
      max_key = sorted(approx_max , reverse = True)[0]
      draw_cnt = approx_max[max_key]

      #print(draw_cnt[0])
      #print(draw_cnt[0][0] ,draw_cnt[1][0], depth_img[depth_img == draw_cnt[0][0]] , len(draw_cnt))

      #print(max_key , cnts_list)
      cv2.drawContours(ero_img, [draw_cnt], -1, (0, 255, 0), 5)
      #cv2.imshow('contour' , ero_img)

      x , y , w , h = cv2.boundingRect(draw_cnt)
      #print(x , y , w , h)

      flatten_arr = mat_arr.copy().flatten()
      depth_avg = 0

      #print(flatten_arr)
      z = 0
      
      for i in range(x , x + w + 1):
            for j in range(y , y + h + 1):
                  z += 1
                  idx = i + j * (x + w)
                  depth_avg = depth_avg + flatten_arr[idx]


      #print('Depth Avg: ' , depth_avg / z)
                  
      cv2.imwrite('src_contour\\' + f[0 : f.index('.')] + '_contour' + '.png', ero_img)
      
##      pil_image = Image.open('depth_100.tiff')
##      crop_image = pil_image.crop((x, y, (w + x), (y + h)))
##      cw , ch = crop_image.size
##      test_img = np.array(list(crop_image.getdata()) , dtype = 'uint8')
##      test_img = np.array(test_img.reshape(cw , ch) , dtype = np.uint16)
##      print(test_img , test_img.shape , test_img.dtype , np.amax(test_img))
##
##      
##      tiff.imsave('src_crop\\' + f[0 : f.index('.')] + '.tif' , test_img)
##      #crop_image.save('depth_crop_test.tiff' , "TIFF")
##      #tiff.imsave('depth_crop_test.tiff' , crop_image)
##      cv2.waitKey(0)
##      cv2.destroyAllWindows()
##
##depth_im = tiff.imread('src_crop\\' + f[0 : f.index('.')] + '.tif')
#####depth_im = (depth_im) & (0xfff8)>> 4
##print(depth_im , depth_im.dtype , depth_im.shape ,np.amax(depth_im))
####cv2.imshow('crop depth' , depth_im)
####cv2.waitKey(0)
####cv2.destroyAllWindows()

