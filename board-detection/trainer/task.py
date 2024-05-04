from trainer.data import Dataset
from trainer.model import LineDetection, Visualisation

from matplotlib import pyplot as plt

dataset = Dataset("./dataset")
linedetection = LineDetection()

img = dataset.get_train_image("c7890b749d14d3488066cbdfac4620fd_jpg.rf.76f2ad22e2a1c25bc8df3e234a2875e4.jpg")
#img = dataset.get_train_image("d494cb268ad7f9f55587de138edc1dc4_jpg.rf.1225ba2bab92010c41fd82111da127c5.jpg")

lines, labels, intersection_points = linedetection.intersections(img)

border_points = linedetection.border(lines, labels, intersection_points)