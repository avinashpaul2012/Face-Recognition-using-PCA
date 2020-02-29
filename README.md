# Face-Recognition-using-PCA
PCA method is used for recognizing faces after finding the eigen faces

In this project, I used the Yale face dataset to implement face recognition by using the Principal component Analysis (PCA) method. 
The training images contain face images of different people, each with different poses and expressions. 
PCA method represents each image in a smaller subspace using dimensionality reduction by using a threshold on eigenvalues and selecting the corresponding eigenvectors.
Every image in the training set is expressed as a linear combination of weights of eigenfaces which is basis for entire training dataset.
For recognizing, I calculated the weights of test image and compared it with weights of the training set. 
The test face with least error between test and training set face was recognised as belonging to the same person. 

