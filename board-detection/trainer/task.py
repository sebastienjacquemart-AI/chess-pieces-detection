from trainer.data import Dataset
from trainer.model import LineDetection, Visualisation

from matplotlib import pyplot as plt

dataset = Dataset("./dataset")
linedetection = LineDetection()
visualisation = Visualisation()

img = dataset.get_train_image("22e74efb18b2d88fba63d25a61bf5f97_jpg.rf.e472c82b49f6f6da28b302bda8ecc4d4.jpg")

edges = linedetection.canny(img, 100, 200)

blur = linedetection.gaussian_blur(img, 3)
blur_edges = linedetection.canny(blur, 100, 200)

lines = linedetection.hough(blur_edges)

labels = linedetection.intersections(lines)
lines_edges = visualisation.lines(img, lines, labels=labels)

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(blur_edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(lines_edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()