from trainer.data import Dataset
from trainer.model import LineDetection

from matplotlib import pyplot as plt

dataset = Dataset("./dataset")
linedetection = LineDetection()

img = dataset.get_train_image("22e74efb18b2d88fba63d25a61bf5f97_jpg.rf.e472c82b49f6f6da28b302bda8ecc4d4.jpg")

edges = linedetection.canny(img)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()