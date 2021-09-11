
import os
import sys
import numpy as np
import cv2
import ImageBlender as IB
import PoissonBlend as PB

def translateImage(image , trans_mat , w , h):

      warp_img = cv2.warpAffine(image , trans_mat , (2 * w , 2 * h))
      return warp_img




def detectFeatures(imgs , path):
      
      sift = cv2.xfeatures2d.SIFT_create()
      #orb = cv2.ORB_create()
      #surf = cv2.xfeatures2d.SURF_create(1000)

      kps = []
      descs = []
      
      for i , img in enumerate(imgs):

            img_gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find matching features using SIFT
            kp , des = sift.detectAndCompute(img_gs , None)
            print('Detected' , len(kp), ' features in image ', i , ' Name: ' , path[i])
            kps.append(kp)
            descs.append(des)


      return kps , descs

def match_flann(desc1, desc2, r_threshold = 0.12):
  'Finds strong corresponding features in the two given vectors.'
  ## Adapted from <http://stackoverflow.com/a/8311498/72470>.

  ## Build a kd-tree from the second feature vector.
  FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
  flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5})

  ## For each feature in desc1, find the two closest ones in desc2.
  (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

  ## Create a mask that indicates if the first-found item is sufficiently
  ## closer than the second-found, to check if the match is robust.
  mask = dist[:,0] / dist[:,1] < r_threshold
  
  ## Only return robust feature pairs.
  idx1  = np.arange(len(desc1)).reshape(idx2.shape[0] , 1)

  print(idx1.shape , idx2[: , 0].shape)
  pairs = np.int32(np.append(idx1, idx2[:,0].reshape(idx1.shape[0] , 1) , axis = 1))
  return pairs[mask]

def findMatches(des1 , des2 , l_ratio = 0.7):

      '''
      # Brute force Match descriptors
      bf = cv2.BFMatcher()
      bf_matches = bf.match(des1 , des2)

      # Sort them in the order of their distance
      bf_matches = sorted(bf_matches, key = lambda x : x.distance)

      matches = bf.knnMatch(des1 , des2 , k = 2)      

      best = []'''

      
      # FLANN based matcher
      FLANN_INDEX_KDTREE = 1
      index_params = dict(algorithm = FLANN_INDEX_KDTREE , trees = 4)
      search_params = dict(check = 50)
      flann = cv2.FlannBasedMatcher(index_params,search_params)

      match_2_1 = flann.knnMatch(des2 ,des1 , k=2)

      match_mask_ratio = [[0 , 0] for i in range(len(match_2_1))]
      match_dict = {}

      for i , (m , n) in enumerate(match_2_1):
            if m.distance < 0.7 * n.distance:
                  match_mask_ratio[i] = [1 , 0]
                  match_dict[m.trainIdx] = m.queryIdx

      best = []

      match_1_2 = flann.knnMatch(des1 , des2 , k = 2)
      match_mask_ratio_1_2 = [[0 , 0] for i in range(len(match_1_2))]

      for i , (m , n) in enumerate(match_1_2):
            if m.distance < 0.6 * n.distance:
                  if m.queryIdx in match_dict and match_dict[m.queryIdx]:
                        best.append(m)
                        match_mask_ratio_1_2[i] = [1 , 0]    

      '''
      # Use lowe's ratio test to keep only the best matches
      for m , n in matches:

            if m.distance < 0.6 * n.distance:
                  best.append(m)'''


      return best

def drawMatchedFeatures(img1 , img2 , dst_pts , src_pts , path1 , path2):

      (h1 , w1) = img1.shape[: 2]
      (h2 , w2) = img2.shape[: 2]

      #draw_img = np.zeros((h1 + h2 , max(w1 , w2) , 3) , dtype = np.uint8)

      print('Drawing between: ' , path1 , ' and ' , path2)


      draw_img = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
      print('Draw shape: ' , draw_img.shape , img2.shape)
      draw_img[:h1, :w1] = img1
      draw_img[:h2, w1:w1+w2] = img2      

      '''

      if img2.shape[0] > img1.shape[0]:
            draw_img[0 : h2 , 0 : draw_img.shape[1]] = img2
            draw_img[h2 : h1 + h2 , 0 : draw_img.shape[1]] = img1
            

      else:
            draw_img[0 : h1 , 0 : draw_img.shape[1]] = img2
            draw_img[h1 : h1 + h2 , 0 : draw_img.shape[1]] = img1

      '''

      # Draw Lines
      for (x1 , y1) , (x2 , y2) in zip(np.int32(dst_pts) , np.int32(src_pts)):
            cv2.line(draw_img, (x1, y1), (x2+w1, y2), (255, 0, 255), lineType=cv2.LINE_AA)
      
      
      return draw_img

def calculateSizeOffset(im1 , im2 , H):

      (h1 , w1) = im1.shape[ : 2]
      (h2 , w2) = im2.shape[ : 2]

      tl = np.array([0 , 0 , 1])
      tr = np.array([w2 , 0 , 1])
      bl = np.array([0 , h2 , 1])
      br = np.array([w2 , h2 , 1])

      # Remap coordinates
      top_left = np.dot(H , tl)
      top_right = np.dot(H , tr)
      bottom_left = np.dot(H , bl)
      bottom_right = np.dot(H , br)

      # Normalize
      #print(top_left , top_left[2])
      top_left = top_left / top_left[-1]
      top_right = top_right / top_right[-1]
      bottom_left = bottom_left / bottom_left[-1]
      bottom_right = bottom_right / bottom_right[-1]

      left_size = int(min(top_left[0] , bottom_left[0] , 0))
      right_size = int(max(top_right[0] , bottom_right[0] , w1))

      W = right_size - left_size

      top_size = int(min(top_left[1] , top_right[1] , 0))
      bottom_size = int(max(bottom_left[1] , bottom_right[1] , h1))

      H = bottom_size - top_size

      size = (W , H)

      # Calculate offset
      X = int(min(top_left[0] , bottom_left[0] , 0))
      Y = int(min(top_left[1] , top_right[1] , 0))

      offset = (-X , -Y)
      
      return size , offset

def createPano(img1 , img2 , H , size , offset , dst_pts , src_pts , i , j):

      (h1 , w1) = img1.shape[ : 2]
      (h2 , w2) = img2.shape[ : 2]

      pano_img = np.zeros((size[1] , size[0] , 3) , dtype = np.uint8)

      (ox , oy) = offset

      #print('Homography matrix before: ' , H)

      trans_mat = np.matrix([[1 , 0.0 , ox],
                             [0 , 1.0 , oy],
                             [0.0 , 0.0 , 1.0]
                             ])

      H = trans_mat * H

      #print('Homography matrix after: ' , H)

      # Draw transformed image 2 on canvas
      cv2.warpPerspective(img2 , H , size , pano_img , cv2.BORDER_REPLICATE )

      #canvas_img = pano_img[oy : h1 + oy , ox : ox + w1].copy()

      pano_img[oy : h1 + oy , ox : ox + w1] = img1

      #cv2.imshow('pano' , canvas_img)
      #cv2.imshow('img1' , img1)

      #new_pano = np.zeros((2 * pano_img.shape[0] + 1 , 2 * pano_img.shape[1]) , dtype = uint8)
      #print(pano_img.shape[1] // 2 , pano_img.shape[1])

      #pano_img1 , pano_img2 = np.split(pano_img , 2)

      #pano_img1 = pano_img[0 : pano_img.shape[0] // 2 , 0 : pano_img.shape[1] // 2]
      #pano_img2 = pano_img[(pano_img.shape[0] // 2) - 1 : pano_img.shape[0] - 1, (pano_img.shape[1] // 2) : pano_img.shape[1]]

      #print(pano_img1.shape , pano_img2.shape)

      #blend_img = pyramidBlending(pano_img1 , pano_img2)
      #mask = 'rgb_data\\mask_grad.png'
      #mask = cv2.imread('rgb_data\\mask_grad.png' , 0)
      #src_mask = np.ones(canvas_img.shape, canvas_img.dtype)
      #blend_img = PB.blend(pano_img , img1 , src_mask)
      #blend_img = IB.weightedAdd(canvas_img , img1 , mask)

      #src_mask = 255 * np.ones(canvas_img.shape, canvas_img.dtype)

      #center = (img1.shape[0] // 2 , img1.shape[1] // 2)
      #blend_img = cv2.seamlessClone(canvas_img, img1, src_mask, center, cv2.NORMAL_CLONE)

      #print(blend_img)
      #cv2.imshow(str(i) , blend_img)
      
      #cv2.imwrite('blend_pano_' + str(i) + '_' + str(j) + '_' + '.jpg' , blend_img)

      return pano_img

def computePanorama(kps , descs , resized_imgs , rgb_list , dirNm , ty):
      
      
      MIN_MATCH = 4

      img_idx = []
      n = len(resized_imgs)

      print(n)

      if n == 1:
            return

      

      H_mats = []
      k = 0

      i_t = 0
      
      for i in range(n - 1):

            img_matches = match_flann(descs[i] , descs[i + 1])

            print('Matching features between : ' , rgb_list[i] , ' and ' , rgb_list[i + 1],
                  ' is : ' , len(img_matches) , ' idx: ' , i , i + 1)


            # Min matches needed is 4
            if len(img_matches) > MIN_MATCH:

                  # Get the matching points
                  kp1 = kps[i]
                  kp2 = kps[i + 1]

                  #print(kp1[i].pt)

                  dst_pts = np.array([kp1[a].pt for (a , b) in img_matches] , np.float32)
                  src_pts = np.array([kp2[b].pt for (a , b) in img_matches] , np.float32)
                  
                  #dst_pts = np.array([kp1[m.queryIdx].pt for m in img_matches] , np.float32)
                  #src_pts = np.array([kp2[m.trainIdx].pt for m in img_matches] , np.float32)

            
            
                  # Draw the matching features between images
                  match_img = drawMatchedFeatures(resized_imgs[i] , resized_imgs[i + 1] , dst_pts , src_pts , rgb_list[i] , rgb_list[i + 1])
            

                  file_nm = 'matching_' + str(i) + '_' + str(i + 1) + '.jpg'
                  os.makedirs('matchers' , exist_ok = True)
                  cv2.imwrite('matchers\\' + file_nm , match_img)


                  # Compute Homography
                  H, mask = cv2.findHomography(src_pts, dst_pts)
                  H_mats.append(H)
                  img_idx.append((resized_imgs[i] , resized_imgs[i + 1]))

                  print('Found Homography between images: ' , rgb_list[i] , ' and ' , rgb_list[i + 1] , ' idx: ' , i , i + 1)

            

                  (size , offset) = calculateSizeOffset(resized_imgs[i] , resized_imgs[i + 1] , H)


                  print('Output Size for : ' , rgb_list[i] , ' and ' , rgb_list[i + 1] , ' is ' , size)

                  print('Offset for : ' , rgb_list[i] , ' and ' , rgb_list[i + 1] , ' is ' , offset)

                  panorama = createPano(resized_imgs[i] , resized_imgs[i + 1] , H , size , offset , dst_pts , src_pts , i , i + 1)

                  os.makedirs(dirNm , exist_ok = True)
                  pano_nm = 'merging_' + ty + '_' + str(i) + '_' + str(i + 1) + '.jpg'
                  cv2.imwrite(dirNm + '\\' + pano_nm , panorama)
                  

                  try:
                        if rgb_list[i].startswith('merging_'):
                        
                              os.remove(rgb_list[i])
                              
                  except OSError:
                        pass

                  #cv2.imwrite(pano_nm , panorama)
                  

            else:
                  mask = cv2.imread('rgb_data\\mask_grad.png' , 0)

                  
                  os.makedirs(dirNm , exist_ok = True)
                  #pb_merge = IB.Laplacian_blending(resized_imgs[i] , resized_imgs[i + 1])

                  pb_merge = IB.merge_images(rgb_list[i] , rgb_list[i + 1] , 'rgb_data\\mask_grad.png')
                  #pb_merge = IB.pyramidBlending(resized_imgs[i] , resized_imgs[i + 1] , rgb_list[i + 1])
                  #pb_merge = PB.blend(resized_imgs[i] , resized_imgs[i + 1] , mask , rgb_list[i])
                  file_nm = 'merging_' + ty + '_' + str(i) + '_' + str(i + 1) + '.jpg'


                  pb_merge = np.array(pb_merge)
                  #cv2.imwrite('merging\\' + pano_nm , panorama)
                  print(pb_merge.shape)
                  try:
                        if rgb_list[i].startswith('merging_'):
                              os.remove(rgb_list[i])
                  except OSError:
                        pass                  

                  cv2.imwrite(dirNm + '\\' + file_nm , pb_merge)
                  i_t = i


      k = i_t + 1
      return k





def getFiles():

      root_r = 'right'
      root_l = 'left'

      # Lambda functions
      read_f = lambda img : cv2.imread(img)
      reSize_f = lambda img : cv2.resize(img , (0 , 0) , fx = 0.5 , fy = 0.5 , interpolation = cv2.INTER_CUBIC)

      
      right_path = os.listdir('right')
      left_path = os.listdir('left')

      right_nm = [os.path.join(root_r , f) for f in right_path if not f.endswith('.db')]           
      left_nm = [os.path.join(root_l , f) for f in left_path if not f.endswith('.db')]

      right = list(map(read_f , right_nm))
      right_resize = list(map(reSize_f , right))

      left = list(map(read_f , left_nm))
      left_resize = list(map(reSize_f , left))


      return left_nm , left_resize , right_nm , right_resize


def makeFullStitch(full_files , full_nm , ctr):

      kps , descs = detectFeatures(full_files , full_nm)
      computePanorama(kps , descs , full_files , full_nm , dirNm = 'mosaic' , ty = str(ctr))
      


def getChunks(path , n):

      for i in range(0 , len(path) , n):
            yield path[i : i + n]
      

if __name__ =='__main__':
      
      root = 'rgb_data'
      path = [f for f in os.listdir(root) if not f.endswith('.py') and not f.endswith('.db') and not f.startswith('mask') and not f.endswith('.png')]

      print(path , len(path))


      '''for ctr , data in enumerate(getChunks(path , 8)):

            rgb_list = [os.path.join(root , f) for f in data]

            # Lambda functions
            read_f = lambda img : cv2.imread(img)
            reSize_f = lambda img : cv2.resize(img , (0 , 0) , fx = 0.4 , fy = 0.4 , interpolation = cv2.INTER_CUBIC)

            images = list(map(read_f , rgb_list))

            resized_imgs = list(map(reSize_f , images))

            n = len(resized_imgs)

            left_resize = resized_imgs[0 : n // 2]
            right_resize = resized_imgs[n // 2 : ]

            left_nm = rgb_list[0 : n // 2]
            right_nm = rgb_list[n // 2 : ]

            # 1) Detect Features using SIFT , SURF  etc
            kps_l , descs_l = detectFeatures(left_resize , left_nm)
            kps_r , descs_r = detectFeatures(right_resize , right_nm)


            print()
            # 2) Find matches between the images
            # Use FLANN or BF and keep the best matches
            # BF Matcher
            computePanorama(kps_l , descs_l , left_resize , left_nm , dirNm = 'left' , ty = 'l')
            computePanorama(kps_r , descs_r , right_resize , right_nm , dirNm = 'right' , ty = 'r')

            
            while len(left_resize) > 1 and len(right_resize) > 1:

                  kps = []
                  descs = []

                  #print('Current Path: ' , cur_path)
                  #prev_sum = cur_sum

                  '''
                  '''
                  root_r = 'right'
                  root_l = 'left'
            
                  right_path = os.listdir('right')
                  left_path = os.listdir('left')

                  right_nm = [os.path.join(root_r , f) for f in right_path if not f.endswith('.db')]
            
                  left_nm = [os.path.join(root_l , f) for f in left_path if not f.endswith('.db')]

                  right = list(map(read_f , right_nm))
                  right_resize = list(map(reSize_f , right))

                  left = list(map(read_f , left_nm))
                  right_resize = list(map(reSize_f , left))'''
                  
                  '''
      
            

                  left_nm , left_resize , right_nm , right_resize = getFiles()

                  print(right_nm , left_nm)
            
            
                  # 1) Detect Features using SIFT , SURF  etc
                  kps_r , descs_r = detectFeatures(right_resize , right_nm)
                  kps_l , descs_l = detectFeatures(left_resize , left_nm)

                  print()
                  # 2) Find matches between the images
                  # Use FLANN or BF and keep the best matches
                  # BF Matcher
                  idx_r = computePanorama(kps_r , descs_r , right_resize , right_nm , dirNm = 'right' , ty = 'r')
                  idx_l = computePanorama(kps_l , descs_l , left_resize , left_nm , dirNm = 'left' , ty = 'l')

                  print('len of r and l: ' , len(right_nm) , len(left_nm))

                  print('idx r and idx l: ' , idx_r , idx_l)

            

                  if idx_r is not None and idx_l is not None:
                        
                        print('to remove: ' , right_nm[idx_r] , ' ' , left_nm[idx_l])

                        try:
                              print('removing: ' , right_nm[idx_r])
                              os.remove(right_nm[idx_r])
                              
                        except OSError:
                              pass

                        try:
                              print('removing: ' , left_nm[idx_l])
                              os.remove(left_nm[idx_l])
                              
                        except OSError:
                              pass

                  left_nm , left_resize , right_nm , right_resize = getFiles()


            if len(left_resize) == 1 and len(right_resize) == 1:
                  
                  left_nm , left_resize , right_nm , right_resize = getFiles()

                  full_files = left_resize + right_resize
                  full_nm = left_nm + right_nm

                  makeFullStitch(full_files , full_nm , ctr)

                  try:
                        print('removing: ' , right_nm[0])
                        os.remove(right_nm[0])
                              
                  except OSError:
                        pass

                  try:
                        print('removing: ' , left_nm[0])
                        os.remove(left_nm[0])
                              
                  except OSError:
                         pass'''                        

                        



'''
cv2.imshow('train' , train_img)
cv2.imshow('warp' , warp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
