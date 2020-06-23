Title of the paper: Ensemble of multiple CNN Classifiers for HSI classification with Superpixel Smoothing

Authors: Sikakollu Prasanth and Ratnakar Dash (Corresponding Author)

Address : National Institute of Technology, Rourkela, India. PIN 769008.

Email : prasanth.sikakollu12@gmail.com (S. Prasanth); ratnakar@nitrkl.ac.in (R. Dash)

This repository proposes a Mean ensemble of four individual CNN classifiers for HSI classification. The output of the mean ensemble classifier is the input for the superpixel smoothing algorithm. This algorithm removes the mis-classified pixels, thereby increasing the classification accuracy.

The entire code is developed by Sikakollu Prasanth during 2019-2020. The size of the code files combined is 26.5KB. 

It requires Keras library installed with either Theano or Tensorflow as backend. It also requires spectral, numpy, scipy, sklearn, skimage libraries to be installed for the code to run successfully. There are no specific hardware requirements.

There are six Python files in this repository. 'main.py' imports the other five files for execution of the code. The five files contains function definitions to accomplish the tasks.

The code can be executed by running 'main.py' file. The output is the prediction map obtained from mean ensemble of four individual classifiers. It is saved as .mat file in data folder.

'data' folder contains color maps for the three datasets. For superpixel smoothing, 'superpixelGeneration.m' file should be executed in MATLAB.
