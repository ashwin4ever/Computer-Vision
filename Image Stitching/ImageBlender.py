import numpy as np
import cv2
import os
from PIL import Image


def stackImages(imgs):

      '''
      params: imgs: list containing images

      Function appends or concatenates images

      '''

      images = [cv2.imread(img , 0) for  img in imgs]

      images = [im[0 : 1080 , 0 : 1920] for im in images]

      ht = sum(im.shape[0] for im in images)
      wd = max(im.shape[1] for im in images)

      res_img = np.zeros((ht , wd) , dtype = np.uint8)

      y = 0

      for im in images:
            h , w = im.shape

            # Attach vertically with width staying constant
            res_img[y : y + h , 0 : w] = im
            y = y + h

      cv2.imwrite('res.png' , res_img)


def merge_images(image1, image2, mask):

    img1 = Image.open(image1)
    #img2 = Image.open(image2)
    mask = Image.open(mask)

    img = Image.open(image2)

    basewidth = img1.size[0]
    hsize = img1.size[1]

    m_width = basewidth
    m_ht = hsize

    m_img = mask.resize((basewidth,hsize), Image.ANTIALIAS)
    m_img.save('mask.png')
    mask = Image.open('mask.png').convert('RGBA')

      
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('src.png')
    img2 = Image.open('src.png').convert('RGB')

    #mask.show()

    height = img1.size[1] + (img2.size[1])
    width = img1.size[0]

    #print(height , width)
    
    #height = img1.size[1]
    #width = img1.size[0]+(img2.size[0]/2)

    #print(height - img1.size[1])

    newImage = Image.new("RGB", (width, height), (0,0,0))
    newImage.paste(img1, (0 , (height - img1.size[1])) , mask)
    #newImage.show()
    #print(newImage.size)
    #newImage.paste(img1, ((width-img1.size[0]),0))
    newImage.paste(img2, (0 ,0) , mask)

    #newImage2 = Image.new("RGB", (width, height), (0,0,0))
    #newImage2.paste(img2, (0 , (height - img1.size[1])))
    #newImage2.paste(img1, (0 , 0))


    #result = Image.blend(newImage , newImage2 , alpha = 0.5)

    return newImage


def pyramidBlending(img1 , img2 , src , num_levels = 6):

      '''

      This function performs pyramid blending of two images

      1) Load the images A and B
      2) Build Laplacian Pyramids LA and LB
      3) Build Gaussian Pyramid GR from selected region R
      4) Form LS from LA and LB using nodes of GR as weights
          LS(i , j) = GR(1 , j ,) * LA(1 , j) + (1 - GR(i , j)) * LB(i , j)
      

      '''

      #img1 = cv2.imread(img1)
      #img2 = cv2.imread(img2)
      
      #img1 = img1[0 : 1080 , 0 : 1920]
      #img2 = img2[0 : 1080 , 0 : 1920]

      img = Image.open(src)

      basewidth = img1.shape[1]
      hsize = img1.shape[0]
      
      
      
      #wpercent = (basewidth/float(img.size[0]))
      #hsize = int((float(img.size[1])*float(wpercent)))
      
      img = img.resize((basewidth,hsize), Image.ANTIALIAS)
      img.save('src.png')

      img2 = cv2.imread('src.png')
      img1 = img1[: , : , :3]
      img2 = img2[:,:,:3]      

      mask_img = np.zeros_like(img1 , dtype = np.float32)
      

      print(img1.shape)
      
      mask_img[: , img1.shape[1] // 2 :] = 1

      mask_copy = np.zeros(img1.shape , dtype = np.uint8)

      mask_copy[: , mask_copy.shape[1] // 2 : ] = 255

      cv2.imwrite('mask.png' , mask_copy)


      g_im1 = img1.copy()
      g_im2 = img2.copy()
      g_mask = mask_img.copy()

      g1 = [g_im1]
      g2 = [g_im2]
      gm = [g_mask]
      
      # Generating Gaussian pyramids
      for i in range(num_levels):

            g_im1 = cv2.pyrDown(g_im1)
            g_im2 = cv2.pyrDown(g_im2)
            g_mask = cv2.pyrDown(g_mask)

            print(i , g_im1.shape , g_im2.shape)

            g1.append(np.float32(g_im1))
            g2.append(np.float32(g_im2))
            gm.append(np.float32(g_mask))            

      '''

      Laplacian Pyramid is formed by the difference between that
      level in Gaussian Pyramid and expanded version
      of its upper level in Gaussian Pyramid
      
      '''

      lap1 = [g1[num_levels - 1]]
      lap2 = [g2[num_levels - 1]]

      lap_mask = [gm[num_levels - 1]]
      
      for i in range(num_levels - 1 , 0 , -1):

           # print(i , cv2.pyrUp(g1[i]).shape)

            size_1 = (g1[i - 1].shape[1] , g1[i - 1].shape[0])
            size_2 = (g2[i - 1].shape[1] , g2[i - 1].shape[0])

            l1 = np.subtract(g1[i - 1] , cv2.pyrUp(g1[i] , dstsize = size_1))
            l2 = np.subtract(g2[i - 1] , cv2.pyrUp(g2[i] , dstsize = size_2))

            lap1.append(l1)
            lap2.append(l2)
            lap_mask.append(gm[i - 1])

      # Add the left and right halves of the Laplacian images in each level
      # Add top and bottom
      LS = []
      LS_hor = []
      for lapA, lapB in zip(lap1, lap2):
            
            rows, cols, dpt = lapA.shape
            lapVert = np.vstack((lapA[0 : rows // 2 ,:], lapB[0 : rows // 2, :]))
            lapHor = np.hstack((lapA[:, 0:cols // 2], lapB[:, cols // 2:]))
    	    
            LS.append(lapVert)
            LS_hor.append(lapHor)


      '''
      # Blend images in each level using the mask
      LS = []
      for la , lb , lm in zip(lap1 , lap2 , lap_mask):

            # LS(i , j) = GR(1 , j ,) * LA(1 , j) + (1 - GR(i , j)) * LB(i , j)
            ls = la * lm + (1.0 - lm) * lb
            LS.append(ls)
      
            
      '''
      # Reconstruct the image
      print()
      ls_ = LS[0]
      ls_h = LS_hor[0]

      print('ls_ shape: ' , ls_.shape , ls_h.shape)

      ls_img = None
      for i in range(1 , num_levels):

            size = (LS[i].shape[1] , LS[i].shape[0])
            size_h  = (LS_hor[i].shape[1] , LS_hor[i].shape[0])
            print('v size: ' , size , 'h size: ' , size_h)

            ls_h = cv2.pyrUp(ls_h , dstsize = size_h)

            ls_ = cv2.pyrUp(ls_ , dstsize = size)
            
            print('v shape: ' , LS[i].shape , ' h shape: ' , LS_hor[i].shape)
            #ls_ = ls_ + LS[i]

            ls_h = cv2.add(ls_h , LS_hor[i])

            ls_ = cv2.add(ls_ , LS[i])


            #ls_ = np.vstack((ls_ , LS[i]))
            

      # Rotate the image
      h_r , w_r = ls_.shape[: 2]

      center = (w_r / 2 , h_r / 2)

      M = cv2.getRotationMatrix2D(center , -180 , 1.0)

      rot_img = cv2.warpAffine(ls_ , M , (w_r , h_r))

      #cv2.imshow('ls' , ls_)

      #return ls_
      #cv2.imwrite('blended.png' , ls_)
      #cv2.imwrite('rotated.png' , rot_img)
      


def Laplacian_blending(img1, img2):
    
    levels = 3
    # generating Gaussian pyramids for both images
    gpImg1 = [img1.astype('float32')]
    gpImg2 = [img2.astype('float32')]
    for i in range(levels):
        img1 = cv2.pyrDown(img1)   # Downsampling using Gaussian filter
        gpImg1.append(img1.astype('float32'))
        img2 = cv2.pyrDown(img2)
        gpImg2.append(img2.astype('float32'))

    # Generating Laplacin pyramids for both images
    lpImg1 = [gpImg1[levels]]
    lpImg2 = [gpImg2[levels]]

    for i in range(levels,0,-1):
        # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
        tmp = cv2.pyrUp(gpImg1[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg1[i-1].shape[1],gpImg1[i-1].shape[0]))
        lpImg1.append(np.subtract(gpImg1[i-1],tmp))

        tmp = cv2.pyrUp(gpImg2[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg2[i-1].shape[1],gpImg2[i-1].shape[0]))
        lpImg2.append(np.subtract(gpImg2[i-1],tmp))

    laplacianList = []
    for lImg1,lImg2 in zip(lpImg1,lpImg2):
        rows,cols = lImg1.shape[ : 2]

        lImg1 = lImg1[ : , : , : 3]
        lImg2 = lImg2[ : , : , : 3]
        

        print(lImg1.shape , lImg2.shape)
        # Merging first and second half of first and second images respectively at each level in pyramid
        mask1 = np.zeros_like(lImg1)
        mask2 = np.zeros_like(lImg2)
        mask1[:, 0:cols// 2] = 1
        mask2[:, cols // 2:] = 1

        print(mask1.shape , mask2.shape)

        tmp1 = np.multiply(lImg1, mask1.astype('float32'))
        tmp2 = np.multiply(lImg2, mask2.astype('float32'))
        tmp = np.add(tmp1, tmp2)
        
        laplacianList.append(tmp)
    
    img_out = laplacianList[0]
    for i in range(1,levels+1):
        img_out = cv2.pyrUp(img_out)   # Upsampling the image and merging with higher resolution level image
        img_out = cv2.resize(img_out, (laplacianList[i].shape[1],laplacianList[i].shape[0]))
        img_out = np.add(img_out, laplacianList[i])
    
    np.clip(img_out, 0, 255, out=img_out)
    return img_out.astype('uint8')


def weightedAdd(img1 , img2 , mask):

      #img1 = cv2.imread(image1)
      #img2 = cv2.imread(image2)

      img = Image.open(img2)

      img1 = Image.open(img1)

      basewidth = img1.size[0]
      hsize = img1.size[1]
      
      
      
      #wpercent = (basewidth/float(img.size[0]))
      #hsize = int((float(img.size[1])*float(wpercent)))
      
      img = img.resize((basewidth,hsize), Image.ANTIALIAS)
      img.save('src.png')

      img2 = cv2.imread('src.png')
      img1 = np.array(img1)
      img1 = img1[: , : , :3]
      img2 = img2[:,:,:3]      
      
      basewidth = img1.shape[1]
      hsize = img1.shape[0]
      
      img = Image.open(mask)
      
      #wpercent = (basewidth/float(img.size[0]))
      #hsize = int((float(img.size[1])*float(wpercent)))
      
      img = img.resize((basewidth,hsize), Image.ANTIALIAS)
      img.save('mask_resize.png')

      
      alpha = cv2.imread('mask_resize.png' , 0).astype(np.float32)

      print(img1.shape , img2.shape , alpha.shape)

      #alpha = cv2.resize(alpha , (0 , 0) , img1.shape , interpolation = cv2.INTER_AREA)

      aB , aG , aR = cv2.split(img1)
      bB , bG , bR = cv2.split(img2)

      b = (aB * (alpha / 255.0)) + (bB * (1.0 - (alpha / 255.0)))
      g = (aG * (alpha / 255.0)) + (bG * (1.0 - (alpha / 255.0)))
      r = (aR * (alpha / 255.0)) + (bR * (1.0 - (alpha / 255.0)))

      output = cv2.merge((b , g , r))


      dst = cv2.addWeighted(img1 , 0.5 , img2 , 0.5 , 0)

      print(dst.shape)
      #cv2.imshow('dst' , dst)
      cv2.imwrite('dst.jpg' , dst)

      return dst

      
      #cv2.imwrite('new.jpg' , output)


if __name__ == '__main__':
      

      path = [os.listdir(f) for f in os.listdir() if not f.endswith('.py') and not f.endswith('.db') and not f.endswith('.jpg') and not f.endswith('.png')]

      img_list = list(filter(lambda f: not f.endswith('.txt') and not f.endswith('.db') , path[0]))

      root = 'data'

      img_list = [os.path.join(root , f) for f in img_list]

      print(img_list)

      for im in img_list:
      
            nm = im[im.index('\\') + 1 : -3]
      
            img = cv2.imread(im)
            img = img[0 : 1080 , 0 : 1920]
      
            cv2.imwrite('rgb_data\\' + nm + 'jpg', img)

      #pyramidBlending('data\\peanut_img_44.png' , 'data\\peanut_img_45.png')
      #new_img = merge_images('rgb_data\\peanut_img_44.jpg' , 'rgb_data\\peanut_img_45.jpg' , 'rgb_data\\mask_grad.png' )

      #img1 = cv2.imread('rgb_data\\peanut_img_42.jpg')
      #img1 = cv2.imread('rgb_data\\peanut_img_42.jpg')
      #img1 = cv2.imread('rgb_data\\peanut_img_42.jpg')
      #dst = weightedAdd('rgb_data\\peanut_img_42.jpg' , 'rgb_data\\peanut_img_43.jpg' , 'rgb_data\\mask_grad.png')
      #cv2.imwrite('merge.jpg' , np.array(new_img))
      #stackImages(['data\\peanut_img_44.png' , 'data\\peanut_img_45.png'])

      img1 = cv2.imread('rgb_data\\peanut_img_42.jpg')
      img2 = cv2.imread('rgb_data\\peanut_img_43.jpg')
      pyramidBlending(img1 , img2 , 'rgb_data\\peanut_img_42.jpg')
