# chess-board-detection
Our method finds the characteristic structures in an image, such as lines and lattice points, and then assesses their locations and shapes based on a scoring function called polyscore (cf. Section 3.4). The values of the polyscore function define the temperature of each point in the heat map. Based on these polyscore values, we can identify components representing a single chessboard

## Detecting straight lines
SLID is an extension of the standard line detector. Its additional objective is to merge all small segments that are nearly collinear into long straight lines. There are several line detectors that can be applied to solve this problem. More satisfactory results can be achieved by the Canny Lines detector [26], which is the detector that we decided to utilize in our method.

Our proposed SLID algorithm consists of three main steps: 1. Boosting: find all possible segments; 2. Grouping: separate segments into groups of nearly collinear segment; 3. Merging: analyze and merge the segments in each group.

### Boosting
One effective method for boosting the detection of line segments is to adaptively adjust the low and high thresholds of the Canny operator based on the gradient magnitude of the input
image.
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
https://xiaohulugo.github.io/papers/CannyLine_Line_Detection_ICIP2015.pdf

https://github.com/Vincentqyw/LineSegmentsDetection

### Linking
The linking function analyzes two segments AB and CD and attempts to determine if they are located and oriented in such manner that they can be treated as a single straight line segment.

### Merging

## Detecting Lattice Points
The LAPS algorithm takes a 21 × 21 matrix whose elements represent pixels as an input. To verify if an (x, y) point in an image is a chessboard lattice point, it utilizes a sub-image
with coordinates ranging from (x −10, y−10) to (x +10, y+10). Then, the algorithm preprocesses this matrix utilizing the following steps: (1) conversion to grayscale, (2) application
of Otsu method altered by Jassim and Altaani [20], (3) application of Canny detector, and (4) binarization. The preprocessed matrix is handled by two modules: (1) a simple geometric detector that recognizes only perfect cases and (2) a neural network for recognizing deformed and distorted patterns. First, for the geometric detector, if the result is positive, we assume that it represents a chessboard lattice point. Otherwise, we utilize the neural network detector because its result definitively determines if the matrix represents a chessboard lattice point.

### Geometric detector
This detector utilizes the following algorithm: (1) add a 1-pixel-width frame of the back-ground color (black) around the input matrix, (2) perform morphological erosion, (3) find all contours and (4) check if the contour resembles a rhomboid

### neural detector

# chess-piece-recognition

# chess-pieces-detection

https://medium.com/@daylenyang/building-chess-id-99afa57326cd (2016)

The first step to identifying chess pieces from a picture of a board is to detect the board and segment it into 64 little squares. The next step is to identify the chess piece on each of the 64 squares.




https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1617&context=eesp (2021)

In the first part, the goal was to separate the chessboard from the surrounding parts of the image, and then adjust the image so the chessboard was perfectly square. The second part involves finding the “grid” created by the 64 squares on the board. The final part searches the image and finds the location and type of any chess pieces present on the board.

In the first part, the goal was to separate the chessboard from the surrounding parts of the image, and then adjust the image so the chessboard was perfectly square. The second part involves finding the “grid” created by the 64 squares on the board. The final part searches the image and finds the location and type of any chess pieces present on the board.


A. BOARD DETECTION. For this part, the goal was to cut out the chessboard from the surrounding image and shift the perspective of the camera, so it appears completely square (as if the camera was directly above the board).

To start, it was necessary to find the four corners of the board. These would allow the board to be cut from the surrounding image and could be used to shift the board. To do this, the image of the board was first scaled down to a smaller resolution since the full-sized image would take far too long, and the added detail was unnecessary for finding a large object like the board. 

After resizing, the board could be converted from a color, RGB image to a simple grayscale one – this is because the actual colors of the board will have no effect on results. 

After resizing the board, the next step was to try and find the contour (the border) of the chessboard itself. 

After finding the four corners of the chessboard, the next step is to cut out the rest of the image and to “correct” the image so the chessboard is completely square. Shifting the image is done through a perspective transformation, or homography. Since we know where the current corners of the chessboard are, and we know where we want them to go (into a square format), so we can find a matrix to transform the current image into the corrected one.

The transformed chessboard is now square, and all the square spaced in the board are equal size. This will make it possible to find the grid defining each position on the board. However, the pieces on the board have also been stretched and distorted – this could make it difficult to recognize them through the transformation. To get around this, the pieces will be found in the original image, and then their coordinates can be shifted into the coordinate grid of the warped image.

B. GRID DETECTION. The goal of grid detection was to end up with a set of vertices for each of the 64 squares in the chessboard. This would allow the image coordinates of each piece to be mapped to the correct square, so that a list of each piece’s location could be created. This part started with the transformed chessboard.

First, Canny Edge detection was used to find a map of all the edges present in the image. This method works by finding the gradient, or change in pixel magnitude, in both the x and y directions for each pixel in the image. The Canny edge detection on its own has a few problems. It detects all locations where pixel magnitudes change greatly, so picks up numbers and words on the side of the board, as well as the chess pieces. These are not needed to find the underlying grid, so need to be removed. The Canny edge detection on its own has a few problems. It detects all locations where pixel magnitudes change greatly, so picks up numbers and words on the side of the board, as well as the chess pieces. These are not needed to find the underlying grid, so need to be removed. 

To get from this collection of edges to an actual grid, a method called the Hough line transform can be used to find the lines along each edge. The Hough transform involves graphing the image in such a way that any given line (such as y = ax +b, although lines are usually represented using θ and ρ) is given a point in the Hough space graph.

The Hough Lines function does a fairly good job finding all the lines. Some of the lines are not as long as they should be, or are drawn at slight angles, but it consistently finds every line making up the chessboard grid. In many cases, multiple Hough lines were drawn next to each other along the same edge. This was easy to eliminate by discarding lines withing a certain distance from each other.

The Hough Lines function does a fairly good job finding all the lines. Some of the lines are not as long as they should be, or are drawn at slight angles, but it consistently finds every line making up the chessboard grid. In many cases, multiple Hough lines were drawn next to each other along the same edge. This was easy to eliminate by discarding lines withing a certain distance from each other.

C. PIECE DETECTION. In many ways the most important part, the goal of Piece Detection was to search an image for chess pieces and return which pieces were present in the image as well as the location of the pieces. 

In the end, an Object Detection model known as YOLO v4 was used. The name stands for “You Only Look Once,” and after training the model, it can run fast enough to detect objects in real time, which is necessary in order for the chess piece detection to be relevant in an actual game of chess. 

Overall, the YOLOv4 model, once trained is capable of object detection with good accuracy at real time speeds. A comparison of YOLOv4 to other object detectors is shown in Figure XIII. The comparison compares accuracy using the AP performance metric with the speed, in frames per second, on a specific set of hardware.

To train the YOLOv4 model, a prebuilt dataset from Roboflow.com was used – a new image set was not created for this project. This included roughly 600 images of a chessboard with prelabeled chess pieces (each piece is labeled with the piece name and a box around it).

Since these images are all taken of situations that might occur in an actual game of chess, certain pieces are naturally more common.

In the full dataset, pawns were easily the most prevalent. This over representation can negatively affect the training, as it can skew the results towards detecting pawns over other pieces. However, since pawns are also the most common in actual games of chess, it could make the detector more accurate with those pieces. The Queen pieces, on the other hand, were very underrepresented in this dataset. If a new dataset were to be created for this project, it preferably would have another few hundred labelled Queens, Kings, and Bishops. Ideally, around a thousand of each type might be more effective, while avoiding overtraining, but this would require many more images and better hardware to train the object detection model – so is outside the scope for this project. 

Rather than rewrite the YOLOv4 model, a prebuilt python repository was used. This can be found at https://github.com/roboflowai/pytorch-YOLOv4. When the training module is run, the previous dataset can be fed in to train the object detection model. 

D. FULL CHESS DETECTION. 

With the first 3 parts working, they need to be combined in order to turn an image of a chessboard into a list of chess pieces and their coordinate locations. To do this, an image is first run through the chessboard detection and perspective transformation. The grid of chess squares is then found with the grid detection. Finally, the chess pieces themselves are found. To determine where each piece is on the grid, the bottom center point (the point centered between the bottom left and bottom right points) was used as the piece’s location on the original image. This was because when looked at from an angle, the tops of the pieces often overlapped or fell into other squares on the grid. By choosing the bottom of the piece, the piece’s location will always be directly on the square the piece is sitting on in reality.  

However, the point selected for the chess piece will be located in the coordinate system of the original image – but the grid of squares is defined in the transformed image. To get around this, the coordinate of the piece can also be transformed using the same transformation matrix onto the transformed image. 

From here it is relatively trivial to compare the piece’s location to the grid to determine the location on the board. Repeating for each detected piece, and the system can return a list of pieces and which of the 64 squares they reside on.  


CONCLUSION & RESULTS. While this project is not perfectly accurate, it displays the potential to be adjusted to consistently determine the locations of chess pieces on a chessboard. The methods to do so run quickly enough to be used to record a chess game in real time. 

Another big method would be to improve detection accuracy for chess piece detection. It is likely that a different object detection model than the YOLO model that was implemented would be able to do so with more accuracy – although there might be a tradeoff with speed. If the program was keeping track of the state of a chess game over multiple moves, it could be made to be much more accurate by assuming all moves are legal (within the bounds of the game). Since any detected chess piece not fitting the game would be known to be false, the next highest probability prediction could be used instead. This would likely increase the accuracy to near 100%. 

Another big method would be to improve detection accuracy for chess piece detection. It is likely that a different object detection model than the YOLO model that was implemented would be able to do so with more accuracy – although there might be a tradeoff with speed. Another big method would be to improve detection accuracy for chess piece detection. It is likely that a different object detection model than the YOLO model that was implemented would be able to do so with more accuracy – although there might be a tradeoff with speed.

Finally, the grid or chessboard detection could be improved. The current method of finding the lines of the chessboard through Hough lines and then deriving the vertices of the squares has many steps – and as a result, many points where failure could occur. 

https://arxiv.org/pdf/2310.04086.pdf (2023)

A. Chessboard Detection The first step is to employ image processing techniques to detect the chessboard and the individual squares; a challenging task even on its own.

Czyzewski et al. introduced an approach based on iterative heat map generation which visualizes the probability of a chessboard being located in a sub-region of the image. After each iteration, the four-sided area of the image containing the highest probability values is cropped and the process is repeated until convergence. While this method involves a great computational overhead, it is able to detect chess boards from images taken from varied angles, with poor quality, and regardless of the state of the actual chessboard (e.g. damaged chessboard with deformed edges), with a 99.6% detection accuracy.

Wölflein and Arandjelovic proposed a chessboard detection method that leveraging the geometric nature of the chessboard, utilizes a RANSAC-based algorithm to iteratively refine the homography matrix and include all the computed intersection points. Their method demonstrated impres-
sive results, since it successfully detected all of the chessboards in their validation dataset. However, it’s worth noting that the dataset only included images with viewing angles within the range of a player’s perspective.

B. Piece Classification Upon detection of the chess board, the next step the aforementioned approaches employ is piece classification.

In Czyzewski et al. they also leverage domain knowledge, to improve piece
classification, by utilizing a chess engine to calculate the most probable piece configurations and cluster-
ing similar figures into groups to deduce formations based on cardinalitie.

given the variation in appearance between chess sets, Wöolflein and Arandjelovic proposed a novel fine-tuning process for their piece classifier to unseen chess sets.

C. Chess dataset A common problem frequently mentioned in literature (Ding, 2016; Mehta, 2020;
Czyzewski et al., 2020; Wölflein and Arandjelovic, 2021) is the lack of a comprehensive chess dataset. This issue hinders not only the ability to fairly evaluate the proposed methods in a common setting but also impedes the deployment of deep learning end-to-end approaches that require a vast amount of data.

The availability of large-scale annotated datasets is critical to the advancement of computer vision research. In this section, we tackle a main issue in the field of chess recognition (i.e. the lack of a comprehensive dataset) by presenting a novel dataset1 specifically designed for this task. 

Method: end-to-end chess recognition

file:///Users/sebastienjacquemart2/Downloads/jimaging-07-00094-v2.pdf


https://arxiv.org/pdf/1708.03898.pdf (2020)

n the following subsections, we first present the algorithm that is executed during each
stage of image processing (Section 3.1). This algorithm consists of three major sub-procedures
that are executed consecutively to locate a chessboard in an image. These sub-procedures are
detecting straight lines (Section 3.2), finding lattice points (Section 3.3), and constructing
heat maps (Section 3.4). When the final stage of the algorithm is completed, it outputs a
cropped image of the chessboard with corrected perspective. This picture is utilized for lo-
cating and classifying chess pieces and generating a description of the board utilizing FEN
notation. This process is detailed in the final subsection of the methods description (Section
3.5).

Github: https://github.com/maciejczyzewski/neural-chessboard?tab=readme-ov-file

A. Single processing stage

The objective of each stage of the proposed algorithm is to find a better approximation of
the chessboard position in an image. 

https://web.stanford.edu/class/cs231a/prev_projects_2016/CS_231A_Final_Report.pdf

https://arxiv.org/pdf/2009.01649.pdf

