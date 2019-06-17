# Work in progress
Network is not fully functional yet. This implementation is part of a thesis on unsupervised state/action space abstraction and simulation on the MDP. 

# SPAIR_pytorch
Pytorch implementation of [Spatially Invariant Unsupervised Object Detection with VAE](https://research.fb.com/wp-content/uploads/2018/12/Spatially-Invariant-Unsupervised-Object-Detection-with-Convolutional-Neural-Networks.pdf?) 

Additional information can be found at [supplementary paper](http://e2crawfo.github.io/pdfs/aaai_2019_supplementary.pdf)

Implementation based on the original tensorflow implementation by Eric Crawford: https://github.com/e2crawfo/auto_yolo 
(Special thanks to Eric for patiently explaining the details of his implementation)

# dependencies

* Numpy
* Pytorch 1.0+
* Python 3.5+ 
* TensorboardX

# Hyperparameters 
`spair/config.py` contains hyperparameters used in the model with comments. 

# Run
`python train.py --gpu` to start training with all available GPUs

# Data
Coming soon... 


