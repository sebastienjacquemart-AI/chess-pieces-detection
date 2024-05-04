from matplotlib import pyplot as plt
from collections import defaultdict

import cv2 as cv
import numpy as np

class LineDetection:
    def __init__(self):
        pass
    
    def intersections(self, img):
        def canny(img, low_tresh, high_tresh):
            edges = cv.Canny(img,low_tresh,high_tresh)
            return edges
        
        def gaussian_blur(img, kernel_size):
            blurred_img = cv.GaussianBlur(img,(kernel_size, kernel_size),0)
            return blurred_img
        
        def hough(edges):
            lines = cv.HoughLines(edges, 2, np.pi/180, 160) #, minLineLength=300, maxLineGap=70)

            assert lines is not None, "no lines found in the image, check hough params"

            return lines
    
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

        blur = gaussian_blur(img, 3)

        blur_edges = canny(blur, 100, 200)

        lines = hough(blur_edges)

        labels = segment_by_angle(lines)
        
        intersections = segmented_intersections(lines, labels)

        visualisation = Visualisation() ### A BIT WEIRD
        lines_edges = visualisation.lines(img, lines, labels=labels, intersections=intersections)


        plt.subplot(221),plt.imshow(img,cmap = 'gray') ### MAKE METHOD
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(blur,cmap = 'gray')
        plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(blur_edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(lines_edges,cmap = 'gray')
        plt.title('Line Image'), plt.xticks([]), plt.yticks([])
        
        plt.show()

        return lines, labels, intersections
    
    def border(self, lines, labels, intersections):
        def convex_hull(points):
            return cv.convexHull(points)
        
        def surface_area(hull):
            x, y, w, h = cv.boundingRect(hull)

            rectangle_area = w * h

            # Calculate area of each triangle and subtract from rectangle area
            triangles_area = 0
            print(len(hull))
            for i in range(len(hull)):
                # Get consecutive points forming the hull
                p1 = hull[i]
                p2 = hull[(i + 1) % len(hull)]

                # Calculate area of triangle formed by hull edge and rectangle
                triangle_area = 0.5 * abs((x*(p1[1] - p2[1]) + p1[0]*(p2[1] - y) + p2[0]*(y - p1[1])))
                triangles_area += triangle_area

            # Subtract area of triangles from rectangle area to get hull surface area
            hull_area = rectangle_area - triangles_area
            return hull_area
        
        def alpha(hull):
            return np.sqrt(surface_area(hull))/7
        
        def centroid(points):
            return

        hull = convex_hull(intersections)
        x,y,w,h = surface_area(hull)


        plt.scatter(intersections[:,0], intersections[:,1]) 
        plt.plot(hull[:, 0, 0], hull[:, 0, 1], color='green', linewidth=2, linestyle='-', label='Convex Hull')

        plt.show()

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
        

class Visualisation:
    def __init__(self):
        pass

    def lines(self, img, lines, labels=None, intersections=None, drawLines=True, drawIntersections=True):
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

        return line_image

    
    def labels(self, pts, labels):
        colors = ['r', 'g', 'b']  # Define colors for each label
        for label_id in range(len(colors)):
            label_indices = np.where(labels == label_id)[0]
            plt.scatter(pts[label_indices, 0], pts[label_indices, 1], c=colors[label_id], label=f'Label {label_id}')
        plt.show()