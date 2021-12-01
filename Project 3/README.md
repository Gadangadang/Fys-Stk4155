### Project 3 Using machine learning to solve the diffusion equation

Before running the code in project 3, please download the required python packages from the requirements.txt file <br />
in the main file. This can be done by typing the following line into the terminal from the appropriate folder:
```
pip install -r requirements
```
It is also good to remember to read the section in the main ReadME file regarding TensorFlow on M1 macs if one has that, <br />
as the software will be used in some capacity in this project. 

This project aimes to try to solve two important problems, PDE's and eigenvalue problems. In project we will look at the diffusion equation. <br />
We first solve the equation by means of explicit Forward Euler discretization. Then we train the network to solve that equation, <br />
using our own loss and train function in synch with the Tensorflow architecture. Then we use the same code to try to compute the eigenvectors of <br />
a matrix A. <br />


The project folder is structured in the following way: <br />
The "article" folder contains figures, animations and the project raport. <br />

The "code" folder contains several subfolders: <br />
* subfolder "tf_checkpoints" contains checkpoints during computation. By default not activated.
* subfolder "tf_models" contains saved models that can be reused later.
* all code for the neural network solver, explicit solver, eigenvalue solver, and timer

