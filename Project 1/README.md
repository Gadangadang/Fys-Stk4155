<h1>Project 1</h1>

Before running the code in project 1, please download the required
python packages from the requirements.txt file in the main file.
This can be done by typing the following line into the terminal
from the appropriate folder:

```
pip install -r requirements
```

This project aims to understand linear regression, focusing on
Ordinary Least Squares, Ridge, and Lasso regression. We first test out
the methods on the Franke Function, given in the project 1 task pdf.
After some analysis we test our methods on a geographic area, in our case,
a part of Saudi Arabia, to determine the best model, and for which
hyperparameters.

The folder is structured as follows:
Functions.py holds the most general and the most used functions.
Files named "Task_{}.py" with letters a to f have functions
more specifically needed for the corresponding task. This is
not however exclusive usage, as some files inherits functions
from previous files. Each function and all its parameters are thoroughly
in the functions themselves.
