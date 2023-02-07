import random
import time
import numpy as np
from pandas import DataFrame
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """

        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        self.probArray = None
        self.yPositive = 0
        self.yNegative = 0
        pass

    def fit(self, X, Y):
        # The length of a singular row in X is the amount of features we are classifying over or in our terms, the number of unique words that are actually being tokenized
        # we need a one dimensional arraya of length D, where each index corresponds to a tuple that looks like this (prob word | +, prob word | -)
        # we update the tuple for each row in X
        probArray = [(0, 0)] * X.shape[1]
        totalWordCountPos = 0
        totalWordCountNeg = 0
        
        for i, sentence in enumerate(X):
            if(Y[i] == 0):
                self.yNegative += 1
            else:
                self.yPositive += 1
            for j, word in enumerate(sentence):
                # we need to come up with the probability
                if(Y[i] == 0):
                    probArray[j] = (probArray[j][0], probArray[j][1] + word)
                    totalWordCountNeg += word
                    
                else:
                    probArray[j] = (probArray[j][0] + word, probArray[j][1])
                    totalWordCountPos += word
                    
                
        for i in range(len(probArray)):
            probArray[i] = (max(probArray[i][0] /totalWordCountPos, (probArray[i][0] + 1)/(totalWordCountPos + len(X[0]))), max(probArray[i][1]/totalWordCountNeg, (probArray[i][1] + 1)/(totalWordCountNeg + len(X[0]))))
            # to prevent divide by zero errors when calculating logs, we create a minimum value for the probability that 'approaches' zero
        self.probArray = probArray
        self.yNegative /= X.shape[0]
        self.yPositive /= X.shape[0]

    def getLabel(self, sentence):
        # returns a predicted boolean value

        sumPos = 0
        sumNeg = 0
        for i, word in enumerate(sentence):
            if(word == 0):
                continue # this word is not in the sentence
            # otherwise, the word is in the sentence
            # figure out the probability of that word
            sumPos += np.log(self.probArray[i][0]) * word
            sumNeg += np.log(self.probArray[i][1]) * word

        positiveBayes = np.log(self.yPositive) + sumPos
        negativeBayes = np.log(self.yNegative) + sumNeg

        return positiveBayes > negativeBayes    


    def predict(self, X):
        # We use the naive bayes formula for negative and for positive
        # We use the log regularization technique on the formula
        # We compare these two values

        # We will be returning a new array of N length, where N is the number of sentences
        # [True, False, False, True, etc]

        # each word in the sentence is tokenized and contains the frequency

        # P(y=pos) * SUM p(each word | pos)
        # --------------------------------------------------------------------------
        # P(y=pos) * SUM p(each word | pos)    +     P(y=neg)*SUM p(each word| neg)

        # log that ^
        returnValue = []
        for sentence in X:
            returnValue.append(self.getLabel(sentence))
        return returnValue


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    # weights = weights + learning rate * (expected - observed) * input
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!

        self.weights = None
        self.epsilon = .000001 # Hardcoded Epsilon Value
        self.alpha = .01
        self.bias = None
        self.lam = .000001

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    def fit(self, X, Y):
        # Add your code here!
        # Stohastic Gradient Descent:
        self.weights = np.zeros(len(X[0]))
        index_collection = list(range(0, X.shape[0] - 1))
        self.bias = np.full(self.weights.shape, 0)
        a = 0
        while True: # while not converged
            index = index_collection[a]
        
            gradient = np.full(self.weights.shape, self.sigmoid(self.weights.dot(X[index]) + self.bias)) - Y[index]  # This the starting gradient value
            gradient *= X[index] * self.alpha
            self.weights = np.add(np.subtract(self.weights, gradient), self.lam * np.square(self.weights))
            # Have we converged?
            if np.linalg.norm(gradient) <= self.epsilon:
                break # we have
            a += 1
            a %= len(index_collection)

        
    
    def predict(self, X):
        # Add your code here!
        ret = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            ret[i] = self.sigmoid(X[i].dot(self.weights + self.bias)) > .5
        return ret


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
