# Project 2 Classification and regression, from linear and logistic regression to neural networks
Before running the code in project 2, please download the required python packages from the requirements.txt file <br />
in the main file. This can be done by typing the following line into the terminal from the appropriate folder:
```
pip install -r requirements
```
It is also good to remember to read the section in the main ReadME file regarding TensorFlow on M1 macs if one has that, <br />
as the software will be used in some capacity in this project. 

This project aimes to look at how Neural Networks handle different challanges, from regression to classification. <br />
We look first at Stochastic gradient descent compared to OLS and Ridge, and then implement that method into <br />
the Neural Network and compare it to OLS and Ridge aswell. Then we will try to permorme classification on breast cancer <br />
and digit data from Sciki-Learn, and test its accuracy against logistic regression and Scikit-learn. 

The project folder is structured in the following way: <br />
The "article" folder contains figures and the project raport. <br />

The "code" folder contains several subfolders: <br />
* "data" contains the raw mnist data from scikit-learn. 
* "Analysis and grid search" contains all code doing analysis on data or gridsearch for hyperparameters
* "FunctionsV2" containts commonly used functions for other scripts
* "Tests and Validation" containts validation and test scripts to ensure that the model works compared to existing software

Outside the subfolders are the two main classes NeuralNetwork.py as SGD.py. They are the main engines for this project <br />
and have documentation inside the constructur. 
