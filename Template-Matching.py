import cv2
import matplotlib.pyplot as plt
import numpy as np

#uploading full image
full = cv2.imread('/home/nlab/OpenCv_Tutorials_Shobhit/Object Detection/group.jpg')
full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)


face = cv2.imread('/home/nlab/OpenCv_Tutorials_Shobhit/Object Detection/solo.jpg')
face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)


#using eval function

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']




#AB HUM EK FOR LOOP CHALAYENGE JO TEMPLATE MATCHING KREGA EVAL FUNCTION KE SATH

for m in methods:

    #create a copy of full image
    full_copy = full.copy()
    method = eval(m)

    #template matching
    res = cv2.matchTemplate(full_copy,face,method)

    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc

    else:
        top_left = max_loc

    height,width,channels = face.shape

    bottom_right = (top_left[0]+width,top_left[1]+height)
    cv2.rectangle(full_copy,top_left,bottom_right,(0,0,255),10)

    #plot and show images

    plt.subplot(121)
    plt.imshow(res)
    plt.title('Heat map of Template Matching')
    #plt.show()

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detection of Template')

    #titlw with the method used
    plt.suptitle(m)

    plt.show()









