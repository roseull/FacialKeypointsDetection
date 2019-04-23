# Facial Keypoints Detection
Find the key points on the human face
### --- Objective ---<br />
The objective of this task is using Convolutional Neural Network(CNN) to predict keypoint positions on face images.

### --- Data description ---<br />
The data is from Kaggle, which contains a set of face images. The input image consists of a list of 96x96 pixels. 
https://www.kaggle.com/c/facial-keypoints-detection

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. 15 keypoints in the dataset represent the different elements of the face (left_eye_center, right_eye_center, nose_tip …);

The input image is given in the last field of the data files, consists of a list of pixels, as integers in (0,255). The images are 96x96 pixels;

The training data contains a list of 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels:<br />  

Test data contains a list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels:<br />  

### --- CNN Model ---<br />
The formula for calculating the output size(height or length) for any given convolutional layer is<br />

<a href="https://www.codecogs.com/eqnedit.php?latex=O=\frac{W&space;-&space;K&space;&plus;&space;2P}{S}&space;&plus;&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O=\frac{W&space;-&space;K&space;&plus;&space;2P}{S}&space;&plus;&space;1" title="O=\frac{W - K + 2P}{S} + 1" /></a>

where O is the output height/length, W is the input height/length, K is the filter size, P is the padding, and S is the stride.<br /><br />

#### The detailed design in CNN model:<br />
![Alt text]( cnn_model.jpg?raw=true "")<br />

The input image in the training set is 96x96 pixels.<br />

The network contains 4 convolutional layer packs and 3 fully connection layers. In the first two conv-layers, the kernels have a size 3x3 with stride 1; In the conv-layers 3 and 4, the kernels have a size 2x2 with stride 1. Each conv-layer is followed by a ReLU activation and a 2x2 max pooling layer with stride 2.<br />

The first convolutional layer contains 32 filters of size(3x3). The output feature map is sized 94x94. After max pooling, the shape of output feature map is 47x47x32.<br />

The second convolutional layer contains 64 filters of size(3x3). The output feature map is sized 45x45. After max pooling, the shape of output feature map is 23x23x64.<br />

The third convolutional layer contains 128 filters of size(2x2). The output feature map is sized 22x22. After max pooling, the shape of output feature map is 11x11x128.<br />

The fourth convolutional layer contains 256 filters of size(2x2). The output feature map is sized 10x10. After max pooling, the shape of output feature map is 5x5x256.<br />

After the 4 convolutional layer packs, 3 fully connected layers are followed. Each layer has 512, 512 and 30 outputs. A dropout are applied on FC2. The final output vector contains regression value of positions of the 15 keypoints(x, y) <br />

### --- Optimizer ---<br />
Stochastic Gradient Descent (SGD) and its variants are probably the most used optimization algorithms for machine learning in general and for deep learning in particular.<br />
![Alt text]( sgd.jpg?raw=true "")<br />
Adam is a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. Adam Optimizer is used in this project.

### --- Evaluation ---<br />
Submissions are scored by Kaggle on the Root Mean Squared Error(RMSE):<br />

<a href="https://www.codecogs.com/eqnedit.php?latex=RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}{(y_i-\hat{y}_i)}^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}{(y_i-\hat{y}_i)}^2}" title="RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}{(y_i-\hat{y}_i)}^2}" /></a>

### --- Future work ---<br />
Feature engineering is needed in the next step.

### --- Result visualization ---<br />
![Alt text]( result.png?raw=true "")<br />
