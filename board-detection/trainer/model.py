from matplotlib import pyplot as plt

import cv2 as cv
import math

class LineDetection:
    def __init__(self):
        pass

    def canny(self, img):
        edges = cv.Canny(img,100,200)
        return edges
    
    def canny_pf(self, img):
        return
    
    def linking_function(self, edges, p, A):
        def get_omega():
            return (math.pi/2)*(1/A**(-4))

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