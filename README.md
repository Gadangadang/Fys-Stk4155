### Fys-Stk4155

This is the project repository for the course Fys-Stk4155. Grade: <br />

Collaborators are Sakarias Frette, Mikkel Metzsch Jensen and William Hirst. <br />
<br />
* Sakarias Frette is doing a master thesis in Computational Physics on Unsupervised Learning on particle collision data. <br />
* William Hirst is doing a master thesis in Computational Physics on Supervised Learning on particle collision data. <br />
* Mikkel Metzch Jensen is doing a master thesis in Computational Material Science on Machine learning on lattice structures. <br />
<br />
Project 1 explores three different linear regression models, OLS, Ridge and Lasso, and studies their accuracy approximating the famous 
Franke Function. Score: 100/100 points <br />
<br />
Project 2 explores neural networks, and compares self made ones to networks from Tensorflow, as well as linear and logistic 
regression models on the Franke Function. Score 94/100 points<br />
<br />
Project 3 explores using neural networks to solve PDE's and eigenvalue problems using Tensorflow infrastructure. It also 
compares against explicit solvers and numerical eigenvalue computation. Score 99/100 points<br />
Extra: Uses boosting, decision trees and neural networks to look at Bias-Variance trade off <br />
for housing data set.  Score 27/30 points <br />

<br />
To install requirements do:

```
pip install -r requirements

```

If one is using the Macbook M1 laptops, (Air or Pro), one needs to take a few more steps 
in order to run tensorflow. Assuming one does not have macOS 12.0+ installed, follow these 
steps here. If one have that version, simple follow the steps listed in the source at the bottom of the page. 

First, remove current conda software, and install conda with miniforge
```
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

Then, install the tensorflow dependencies with conda:
```
conda install -c apple tensorflow-deps
```

Assuming that one has previous versions of tensorflow for mac and metal for gpu,
run the commands below. If not, skip this step:
```
# uninstall existing tensorflow-macos and tensorflow-metal
python -m pip uninstall tensorflow-macos
python -m pip uninstall tensorflow-metal
# Upgrade tensorflow-deps
conda install -c apple tensorflow-deps --force-reinstall
```

If one wants a specific tensorflow version, download version by specifying:

```
conda install -c apple tensorflow-deps==2.6.0
```
here using the example of version 2.6.0.

Now, all we need to do is to install the MacOS tensorflow package and the GPU runner, i.e Metal.
Here it is important to download the correct Metal version, as the newest one requires the MacOS 12.0+
operating system update. Version 0.1.2 is stable.

```
python -m pip install tensorflow-macos
python -m pip install tensorflow-macos==0.1.2
```

Having done this correctly, one should be able to run Tensorflow with ease on a M1 MacBook.


Source: https://developer.apple.com/metal/tensorflow-plugin/
















