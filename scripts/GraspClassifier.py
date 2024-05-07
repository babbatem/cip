'''
Copyright, 2022, Matt Corsaro, matthew_corsaro@brown.edu
'''

import numpy as np
import os
import sys

#pip install gpy
import GPy

# Class that trains a Gaussian Process classifier with the train_buffer_size latest examples available.
class GraspClassifier(object):
    def __init__(self, train_buffer_size=30):
        super(GraspClassifier, self).__init__()

        self.train_buffer_size = train_buffer_size

        self.training_examples = []
        self.training_binary_labels = []

        self.clf = None

    def addBinaryLabeledExample(self, example, label):
        self.training_examples.append(example)
        self.training_binary_labels.append(label)
        if len(self.training_examples) > self.train_buffer_size:
            self.training_examples.pop(0)
            self.training_binary_labels.pop(0)
        assert(len(self.training_examples) == len(self.training_binary_labels))

    def trainClassifier(self, initial_kernel_variance=1.0, initial_kernel_lengthscale=1.0):
        training_data = np.array(self.training_examples)
        training_labels = np.array(self.training_binary_labels).reshape((-1, 1))
        print("Training classifier with data of shape", training_data.shape)
        kernel = GPy.kern.RBF(training_data.shape[1], variance=initial_kernel_variance, lengthscale=initial_kernel_lengthscale)
        num_param_iter = 50#5000
        max_optim_iter = 1000#100000
        # https://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/basic_classification.ipynb
        # https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/classification.html
        lik = GPy.likelihoods.Bernoulli()
        self.clf = GPy.core.GP(X=training_data,
                               Y=training_labels,
                               kernel=kernel,
                               inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                               likelihood=lik)
        for i in range(num_param_iter):
            self.clf.optimize('bfgs', max_iters=max_optim_iter) #first runs EP and then optimizes the kernel parameters

    def predictSuccessProbabilities(self, examples):
        test_data = np.array(examples)
        print("Evaluating examples with shape", test_data.shape)
        # (#, 1) array of positive label probabilities, reshape to vector
        pred_scores = (self.clf.predict(test_data)[0]).reshape(-1)
        return pred_scores

def test():
    gc = GraspClassifier()
    test_poses = []
    for i in range(40):
        test_poses.append(np.random.random((7)).tolist())
        train_pose = np.random.random((7)).tolist()
        train_label = bool((np.random.random((1)) > 0.5)[0])
        gc.addBinaryLabeledExample(train_pose, train_label)
    print("test_poses is a list of len({}), where each element is list of len({}).".format(len(test_poses), len(test_poses[0])))
    gc.trainClassifier()
    test_probs = gc.predictSuccessProbabilities(test_poses)
    print("Predicted probabilities of shape", test_probs.shape)
    print(test_probs)

if __name__ == '__main__':
    test()