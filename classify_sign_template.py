import cv2
import numpy as np
import random as rng

WARPED_XSIZE = 200
WARPED_YSIZE = 300

canny_thresh = 120;

#############


#Open CV 3 Python 2

############
VERY_LARGE_VALUE = 100000

NO_MATCH            =  0
STOP_SIGN           =  1
SPEED_LIMIT_40_SIGN =  2
SPEED_LIMIT_80_SIGN =  3
SPEED_LIMIT_100_SIGN = 4
YIELD_SIGN           = 5 

def show_image_simple(image):
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

font = cv2.FONT_HERSHEY_COMPLEX
def write_on_image(image, text):
    fontFace = cv2.FONT_HERSHEY_DUPLEX;
    fontScale = 2.0
    thickness = 3
    textOrg = (10, 130)
    cv2.putText(image, text, textOrg, fontFace, fontScale, thickness, 8);
    return image

def identify():
    image = cv2.imread("speedsign3.jpg")
    #image = cv2.imread("speedsign4.jpg")
    #image = cv2.imread("speedsign14.jpg")
    #image = cv2.imread("speedsign16.jpg")
    #image = cv2.imread("yield_sign1.jpg")
    image = cv2.imread("stop4.jpg")
    forty_template = cv2.imread("speed_40.bmp")
    print('forty template',forty_template.shape)
    forty_template = cv2.cvtColor(forty_template, cv2.COLOR_BGR2GRAY )
    eighty_template = cv2.imread("speed_80.bmp")
    print('eighty template',eighty_template.shape)
    eighty_template = cv2.cvtColor(eighty_template, cv2.COLOR_BGR2GRAY )
    one_hundred_template = cv2.imread("speed_100.bmp")
    print('one hundred template',one_hundred_template.shape)
    one_hundred_template = cv2.cvtColor(one_hundred_template, cv2.COLOR_BGR2GRAY )

    print("Reading forty template")
    show_image_simple(eighty_template)
    print("Reading eighty template")
    show_image_simple(forty_template)
    print("Reading one hundred template")
    show_image_simple(one_hundred_template)

    image_original = image.copy()
    image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    image_bw = image.copy()
    image = cv2.blur( image_bw, (3,3) )
    sign_recog_result = NO_MATCH


    
    edges = cv2.Canny(image,canny_thresh , canny_thresh*2 )
    im2,contours, hierarchy =  cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE ) # all the contours in the canny image
  
   
      
  
    cnt = contours[0]
    max_area = None
    for d in contours:      # iterate through all contours 
        if cv2.contourArea(d) > max_area: # find the largest contour
            cnt = d
            max_area = cv2.contourArea(d)
    epsilon = 0.02*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    cv2.drawContours(im2, [approx], -1, (0, 225, 0), 3) 

    show_image_simple(im2)

    print approx
    if len(approx) ==4: # a speed sign
        rect1 = approx.reshape (4,2)
        rect = np.zeros((4,2) ,np.float32)
        
        s = rect1.sum(axis =1)
        rect[0] = rect1[np.argmin(s)]
        rect[2] = rect1[np.argmax(s)]
        diff = np.diff(rect1 , axis = 1)
        rect[1] = rect1[np.argmin(diff)]
        rect[3] =  rect1[np.argmax(diff)]
        
        

        dst = np.array([[0, 0],
                      [WARPED_XSIZE, 0],
                      [WARPED_XSIZE, WARPED_YSIZE],
                      [0, WARPED_YSIZE]], np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_bw, M, (WARPED_XSIZE, WARPED_YSIZE))
        
        # find which image matches with the template   
        match = cv2.matchTemplate(warped,forty_template, cv2.TM_CCOEFF_NORMED)
        if (match >=0.85): # very close to a match
            sign_recog_result = forty_template

        match = cv2.matchTemplate(warped,eighty_template, cv2.TM_CCOEFF_NORMED)
        if (match >=0.85):
            sign_recog_result = SPEED_LIMIT_80_SIGN


        match = cv2.matchTemplate(warped,one_hundred_template, cv2.TM_CCOEFF_NORMED)
        if (match >=0.85):
            sign_recog_result = SPEED_LIMIT_100_SIGN
            
       
    
    elif len (approx) == 3: #yield sign
        sign_recog_result = YIELD_SIGN
        
       
    elif len(approx) == 8: #stop sign
        sign_recog_result = STOP_SIGN
      
    else: # no match for 70 speed sign
        sign_recog_result = NO_MATCH
       


    show_image_simple(image_original)
 

    if sign_recog_result == NO_MATCH:
        sign_string = "No match"
        file_string = "No_match.jpg"
    elif sign_recog_result == STOP_SIGN:
        sign_string = "Stop sign"
        file_string = "Stop_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_40_SIGN:
        sign_string = "40_sign"
        file_string = "40_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_80_SIGN:
        sign_string = "80_sign"
        file_string = "80_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_100_SIGN:
        sign_string = "100_sign"
        file_string = "100_sign.jpg"
    elif sign_recog_result == YIELD_SIGN:
        sign_string = "Yield sign"
        file_string = "Yield_sign.jpg"

    # save the results
    print(sign_string)
    result = write_on_image(image_original, sign_string)
    cv2.imwrite(file_string, result)


identify()
