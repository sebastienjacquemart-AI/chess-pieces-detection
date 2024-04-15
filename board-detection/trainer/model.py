from matplotlib import pyplot as plt

import cv2 as cv
import numpy as np

class LineDetection:
    def __init__(self):
        pass

    def gaussian_blur(self, img, kernel_size):
        blurred_img = cv.GaussianBlur(img,(kernel_size, kernel_size),0)
        return blurred_img

    def canny(self, img, low_tresh, high_tresh):
        edges = cv.Canny(img,low_tresh,high_tresh)
        return edges
    
    def canny_pf(self, img):
        return
    
    def hough(self, img, edges):
        line_image = np.copy(img) * 0

        lines = cv.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=300, maxLineGap=70)

        assert lines is not None, "no lines found in the image, check hough params"
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        
        lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)

        return lines_edges
    
    def get_intersections():
        return
    
    def linking_function(self, edges, p, A):
        def get_omega():
            return (np.pi/2)*(1/A**(-4))

        def get_t_delta():
            return p*get_omega()
        
        def get_delta(a,b):
            return (a+b)*get_t_delta()
        
        def merge_segments():

            return
        

class Plot:
    def __init__(self):
        pass

    def plot_image(self, img, title):
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        plt.show()