import cv2 as cv
import numpy as np
import math

#Open CV 3 Python 2
def main():
    fname = 'track.jpg'
    src = cv.imread(fname, cv.IMREAD_GRAYSCALE)

    canny = cv.Canny(src, 120, 250, None, 3)
    c = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

    HoughLines = cv.HoughLines(canny, 1, np.pi / 180, 200, None, 0, 0)
    
    if HoughLines is not None:
        for i in range(0, len(HoughLines)):
            rho = HoughLines[i][0][0]
            theta = HoughLines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(c, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
         
    
    cv.imshow('Hough Line Transformation', c)


    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    print cv.__version__
    main()
   
