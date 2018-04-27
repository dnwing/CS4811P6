# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        Plabel = util.Counter() 
        featLabel = util.Counter()
        conditionalProb = util.Counter()
        bestParams = util.Counter()
        myAccuracy = -1       
        
        for i in range(len(trainingData)):
            data = trainingData[i]
            label = trainingLabels[i]
            Plabel[label] += 1
            for feat, val in data.items():
                featLabel[(feat,label)] += 1
                if val > 0:
                    conditionalProb[(feat,label)] += 1
        
        for k in kgrid:
            prob = util.Counter()
            counts = util.Counter()
            condProb = util.Counter()
            
            for key, val in Plabel.items():
                prob[key] += val
            for key, val in featLabel.items():
                counts[key] += val
            for key, val in conditionalProb.items():
                condProb[key] += val
                
            for label in self.legalLabels:
                for feat in self.features:
                    condProb[(feat,label)] += k
                    counts[(feat,label)] += 2*k
                    
            prob.normalize()
            for param, count in condProb.items():
                condProb[param] = count*1.0 / counts[param]
                
            self.pLabel = prob
            self.condProb = condProb
            
            predict = self.classify(validationData)
            correct = [predict[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            print correct
            
            if correct > myAccuracy:
                bestParams = (prob, condProb, k)
                myAccuracy = correct
                
        self.pLabel, self.condProb, self.k = bestParams
        
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()
        for label in self.legalLabels:
            logJoint[label] = math.log(self.pLabel[label])
            for feat, val in datum.items():
                if val > 0:
                    logJoint[label] += math.log(self.condProb[feat,label])
                else:
                    logJoint[label] += math.log(1-self.condProb[feat,label])
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []
        
        for feat in self.features:
            featuresOdds.append((self.condProb[feat,label1]/self.condProb[feat,label2], feat))
            featuresOdds.sort()
            for val, feat in featuresOdds[-100:]:
                featureOdds = feat
        
        return featuresOdds
