# Image_classifier
Development and implementation of an image classifier model for classifying flowers from the Oxford 102 dataset.
Transfer learning was implemented using the MobileNet pre-trained model.

## Installations
The Anaconda distribution is required to run the code in this repository. 
The necessary libraries to run both the python notebook and the python script are: 

- numpy: v 1.23.5
- tensorflow: v 2.12
- keras: v 2.12

The version of python used for this task is Python 3.8.5. 

## Motivation
This model was developed as part of the second project of the Introduction to Machine Learning Udacity Nanodegree

## Files in this repository
This repository contains a notebook and python script.

### Project_Image_Classifier_Project.ipynb
- Model development was conducted using this notebook

### predict.py
- The command line app that takes in an image and the trained model, then makes a prediction of the flower type.

Options: 
- --category_names - when this option is set, reads in a json file mapping labels to flower names.
- --top_k - when this option is set, prints out the requested number of classes. Default is 5
- Example usage: python predict.py ./test_images/wild_pansy.jpg my_model.h5 --category_names label_map.json --top_k 3
