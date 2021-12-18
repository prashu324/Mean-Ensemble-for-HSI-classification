Title of the paper: Ensemble of multiple CNN Classifiers for HSI classification with Superpixel Smoothing
https://www.sciencedirect.com/science/article/pii/S0098300421001047?via%3Dihub

Authors: Sikakollu Prasanth and Ratnakar Dash (Corresponding Author)

Address : National Institute of Technology, Rourkela, India. PIN 769008.

Email : prasanth.sikakollu12@gmail.com (S. Prasanth); ratnakar@nitrkl.ac.in (R. Dash)

Highlights : 

1. Proposed an Ensemble of multiple CNNs for HSI classification using optimal strategy.

2. Suggested a new idea for optimizing the number of features to be considered for classification.

3. Proposed a superpixel smoothing algorithm to rectify the misclassified pixels at the output of mean ensemble classifier.

Abstract:

Hyperspectral Image analysis has gained much attention due to the presence of rich spectral information. Hyperspectral Image (HSI) classification is being utilized in a wide range of applications. Convolutional Neural Networks (CNN) are popularly used in the image classification tasks due to their capability of extracting spatial features from the raw image data. Creating an ensemble of multiple classifiers generates more robust and reliable classification results. In this paper, we propose an ensemble of four CNN classifiers with superpixel smoothing for the task of HSI classification. Stacked Auto-encoder is utilized to reduce the dimensionality of the hyperspectral data. A new method is suggested to derive the optimal number of features by exploiting the diversity among the classifiers. The uniform Local Binary Patterns (ULBP) are extracted from the HSI and is used along with reduced HSI data for classification. The two single-channel models take reduced HSI cubes as input. The two dual-Channel CNN models explore both ULBP patterns and HSI data simultaneously. We explore various techniques for combining the predictions of individual classifiers and choose the best one for ensembling purpose. The obtained prediction map is made to undergo superpixel based smoothing to remove most of the misclassified pixels. Experimental results on standard data sets confirm the superiority of the proposed ensemble model over the state of the art models. The advantages of superpixel smoothing after CNN classifications are also validated through numerical results and corresponding classification maps.

Implementation:

This repository proposes a Mean ensemble of four individual CNN classifiers for HSI classification. The output of the mean ensemble classifier is the input for the superpixel smoothing algorithm. This algorithm removes the mis-classified pixels, thereby increasing the classification accuracy.

The entire code is developed by Sikakollu Prasanth during 2019-2020. The size of the code files combined is 26.5KB. 

It requires Keras library installed with either Theano or Tensorflow as backend. It also requires spectral, numpy, scipy, sklearn, skimage libraries to be installed for the code to run successfully. There are no specific hardware requirements.

There are six Python files in this repository. 'main.py' imports the other five files for execution of the code. The five files contains function definitions to accomplish the tasks.

The code can be executed by running 'main.py' file. The output is the prediction map obtained from mean ensemble of four individual classifiers. It is saved as .mat file in data folder.

'data' folder contains color maps for the three datasets. For superpixel smoothing, 'superpixelGeneration.m' file should be executed in MATLAB.
