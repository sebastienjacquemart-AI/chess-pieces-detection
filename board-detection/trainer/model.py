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
        lines = cv.HoughLines(edges, 2, np.pi/180, 160) #, minLineLength=300, maxLineGap=70)

        assert lines is not None, "no lines found in the image, check hough params"

        return lines
    
    def intersections(self, lines):
        def segment_by_angle(lines):
            # returns angles in [0, pi] in radians
            angles = np.array([line[0][1] for line in lines])
            # multiply the angles by two and find coordinates of that angle
            pts = np.array([[np.abs(np.cos(angle)), np.abs(np.sin(angle))]
                            for angle in angles], dtype=np.float32)

            # run kmeans on the coords
            default_criteria_type = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
            labels, centers = cv.kmeans(pts, 2, None, (default_criteria_type, 10, 1.0), cv.KMEANS_RANDOM_CENTERS, 10)[1:]
            labels = labels.reshape(-1)  # transpose to row ve

            colors = ['r', 'g', 'b']  ### MAKE METHOD ###
            for label_id in range(len(colors)):
                label_indices = np.where(labels == label_id)[0]
                plt.scatter(pts[label_indices, 0], pts[label_indices, 1], c=colors[label_id], label=f'Label {label_id}')
            plt.show()

            return labels
        
        def intersection(line1, line2):
            """Finds the intersection of two lines given in Hesse normal form.

            Returns closest integer pixel locations.
            See https://stackoverflow.com/a/383527/5087436
            """
            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            return (x0, y0)


        def segmented_intersections(lines, labels):
            """Finds the intersections between groups of lines."""
            lines_group1 = []
            lines_group2 = []

            for line, label in zip(lines, labels):
                if label == 0:
                    lines_group1.append(line)
                elif label == 1:
                    lines_group2.append(line)
                else:
                    continue
            
            intersections = []
            for line1 in lines_group1:
                for line2 in lines_group2:
                    intersections.append(intersection(line1, line2))

            return np.array(intersections)
        
        labels = segment_by_angle(lines)
        intersections = segmented_intersections(lines, labels)

        intersections_x = [x[0] for x in intersections] #### MAKE METHOD
        intersections_y = [x[1] for x in intersections]

        plt.scatter(intersections_x, intersections_y) 
        plt.show()

        return labels, intersections
    
    def hull(self, points):
        hull = cv.convexHull(points)



        return hull
    


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

    def lines(self, img, lines, hull, labels=None, intersections=None, drawLines=True, drawIntersections=True, drawHull=True):
        gray = np.copy(img)
        line_image = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

        color_dict = {
            0: (255, 0, 0),
            1: (0, 255, 0),
        }

        if labels is None:
            labels = [1] * len(lines)

        # Draw lines
        if drawLines:
            for line, label in zip(lines, labels):
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho

                    length = 2000
                    x1 = int(x0 + length * (-b))
                    y1 = int(y0 + length * (a))
                    x2 = int(x0 - length * (-b))
                    y2 = int(y0 - length * (a))

                    cv.line(line_image, (x1, y1), (x2, y2), color_dict[label], 5)

        # Draw intersections if provided
        if intersections is not None and drawIntersections:
            for intersection in intersections:
                cv.circle(line_image, (int(intersection[0]), int(intersection[1])), 2, (0, 0, 255), -1)

        if hull is not None and drawHull:
            cv.polylines(line_image, [hull], True, (0, 255, 0), 2)

        return line_image

    
    def labels(self, pts, labels):
        colors = ['r', 'g', 'b']  # Define colors for each label
        for label_id in range(len(colors)):
            label_indices = np.where(labels == label_id)[0]
            plt.scatter(pts[label_indices, 0], pts[label_indices, 1], c=colors[label_id], label=f'Label {label_id}')
        plt.show()