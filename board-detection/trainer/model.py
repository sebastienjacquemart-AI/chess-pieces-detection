from matplotlib import pyplot as plt
from collections import defaultdict

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

    def hough(self, edges):
        lines = cv.HoughLines(edges, 2, np.pi/180, 200) #, minLineLength=300, maxLineGap=70)

        assert lines is not None, "no lines found in the image, check hough params"

        return lines
    
    def intersections(self, lines):
        def segment_by_angle(lines):
            # returns angles in [0, pi] in radians
            angles = np.array([line[0][1] for line in lines])
            # multiply the angles by two and find coordinates of that angle
            pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                            for angle in angles], dtype=np.float32)

            # run kmeans on the coords
            default_criteria_type = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
            labels, centers = cv.kmeans(pts, 2, None, (default_criteria_type, 10, 1.0), cv.KMEANS_RANDOM_CENTERS, 10)[1:]
            labels = labels.reshape(-1)  # transpose to row ve

            return labels
        
        segmented = segment_by_angle(lines)

        return segmented
    
    def linking_function(self, edges, p, A):
        def get_omega():
            return (np.pi/2)*(1/A**(-4))

        def get_t_delta():
            return p*get_omega()
        
        def get_delta(a,b):
            return (a+b)*get_t_delta()
        
        def merge_segments():

            return
        

class Visualisation:
    def __init__(self):
        pass

    def lines(self, img, lines, labels=None):
        gray = np.copy(img)
        line_image = cv.cvtColor(gray,cv.COLOR_GRAY2RGB)

        color_dict = {
            0: (255,0,0),
            1: (0,255,0),
            2: (0,0,255)
        }

        if labels is None:
            labels = [1] * len(lines)

        for line, label in zip(lines, labels):
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                length = 1000
                x1 = int(x0 + length * (-b))
                y1 = int(y0 + length * (a))
                x2 = int(x0 - length * (-b))
                y2 = int(y0 - length * (a))

                cv.line(line_image,(x1,y1),(x2,y2),color_dict[label],5)
        
        #lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)

        #cv.imshow("skt",line_image)

        return line_image
    
    def labels(self, pts, labels):
        colors = ['r', 'g', 'b']  # Define colors for each label
        for label_id in range(len(colors)):
            label_indices = np.where(labels == label_id)[0]
            plt.scatter(pts[label_indices, 0], pts[label_indices, 1], c=colors[label_id], label=f'Label {label_id}')
        plt.show()