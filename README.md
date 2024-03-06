# CM3070 Final Project
## Template CM3015 Machine Learning and Neural Networks - Project Idea Title 1: Deep Learning on a public dataset
This repository is about the CM3015 Machine Learning and Neural Networks - Project Idea Title 1: Deep Learning on a public dataset. The dataset chosen can be found [here](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and is part of the M5 Forecasting - Accuracy challenge hosted on Kaggle. The question the organizers of the challenge were posing is „Can you estimate, as precisely as possible, the point forecasts of the unit sales of various products sold in the USA by Walmart?“


## Structure
The repository is broken down into an environment folder and a source folder containing the model ipynb's.
- The data preprocessing and preparation will be conducted by running the Final_Data_Prep.ipynb file
- The Final_Model_V8.ipynb file can be used for further keras tuner runs
- The resulting hyperparameter configurations can than be stored and manually adjusted in the src/models/configs/config.csv file
- The file Final_Model_V9.ipynb will then be used to read in the configuration from the config.csv file and for the final training and prediction
- The file src/submissions/sample_submission.csv will then be created and can be used to submit on Kaggle
