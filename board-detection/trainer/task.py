from trainer.data import Dataset
from trainer.model import LineDetection, Visualisation

from matplotlib import pyplot as plt

dataset = Dataset("./dataset")
linedetection = LineDetection()
visualisation = Visualisation()

img = dataset.get_train_image("c7890b749d14d3488066cbdfac4620fd_jpg.rf.76f2ad22e2a1c25bc8df3e234a2875e4.jpg")

edges = linedetection.canny(img, 100, 200)

blur = linedetection.gaussian_blur(img, 3)
blur_edges = linedetection.canny(blur, 100, 200)
lines = linedetection.hough(blur_edges)

labels, intersections = linedetection.intersections(lines)
hull = linedetection.hull(intersections)

lines_edges = visualisation.lines(img, lines, hull, labels=labels, intersections=intersections, drawLines=False)


plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(blur_edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(lines_edges)
plt.title('Line Image'), plt.xticks([]), plt.yticks([])
 
plt.show()
