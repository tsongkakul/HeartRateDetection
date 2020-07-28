# Project 3 - Covolutional Neural Network for MNIST classification
- The goal of this homework is for you to become familiar with a deep learning framework.
- You will be asked to design and implement a CNN for MNIST classification.

## Recommend Environment
- [Python3](https://www.python.org/download/releases/3.0/)
- One of the deep learning frameworks

## Dataset
- Download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/).
- Use only the training images for training (You can split it into training set and validation set).
- The testing set will be used for evaluation. **Please only use the testing set download from the above link**. If you decide to import the data using some libraries, please make sure the testing set contains the same samples as the set downloaded from above link.

## Assignment Description
- Design a CNN for MNIST classification. You can use a simple CNN to reduce the computational complexity, since the final performance is not important here.
- Implement the CNN use one of deep learning frameworks.
- Apply cross-validation to select hyparameters including learning rate and batch size. You can choose one of several strategies described in Section 11.4 of Goodfellow's *Deep Learning*. The range of each hyparameter is decided by yourself. Also, one fold cross-validation (use a single validation set) can be used to reduce the time complexity. However, the validation set has to come from the training set.
- Create plots to visualize the cross-validation.
- After obtaining a good set of hyparamters, apply different activation functions including tanh, sigmoid and Relu and compare the performance.
- Visualize the training processing of different activation functions.
- Apply the best trained model or ensemble of models on testing set and save the results.


## Deliverable
- Report should includes (Use [this](https://www.ieee.org/conferences/publishing/templates.html) template): 
1. Structure of the network;
2. The range of each hyperparameters and the set of hyperparameters you choose for experiments.
3. Visualization of cross-validation for hyperparameters selection;
4. Learning curve (loss and accuracy) on both training and validation set for different activation function; 
5. Accuracy on testing set;
6. Analysis of your experiment, including hyperparameters selection and activation function selection;
7. Write up including figures (but no references) should be no more than 6 pages long.

- Source code
- The best results on testing set saved in `csv` file with file name `mnist.csv`. The submission is the same as homework 2. Sample submission of homework 2 can be found [here](../../docs/samples/sample_afs/). **Note this time the result `cvs` file will be `mnist.csv`.**
- Put the source code, result and a `name.csv` file containing your unityID in your AFS space under folder **`proj03`**.
- **Do not shuffle the testing data for submission and testing set contains 10k samples**.
- Make sure to follow this format or you may lose all the credits. Please read the instructions in the sample submission folder carefully. Also, please do not try to modify the `eval.py` to make the testing correct.

## Evaluation for Credits
- We will evaluate the project automatically. So please make sure to follow the submission format as mentioned above.
- The evaluation script can be found [here](../../src/eval/eval.py). Please follow the [instruction](https://github.ncsu.edu/qge2/ece542-2018fall/tree/master/src/eval#evaluation-script) to evaluate your result before submission. Make sure you can get correct output using the script or you may lose all the credits.








