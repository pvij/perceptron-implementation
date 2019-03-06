import csv
import matplotlib.pyplot as plt
import numpy as np


class perceptron:

    def __init__(self, csvFilePath):
        self.csvFilePath = csvFilePath
        self.learningRate = 0.1

    def start(self):
        self.readCSV()
        self.getPositiveAndNegativePoints()
        self.initializeWeightsVector()
        self.applyPerceptronRule()

    def plotData(self):
        plt.scatter(self.x1Positive, self.x2Positive, color="blue")
        plt.scatter(self.x1Negative, self.x2Negative, color="red")

    def abline(self, slope, intercept):
        #    """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    def getPositiveAndNegativePoints(self):
        self.x1Positive = []
        self.x2Positive = []
        self.x1Negative = []
        self.x2Negative = []
        for i in range(self.noOfObs):
            if self.dependentValuesAsVector[i] == 1:
                self.x1Positive.append(self.independentDataSet[i][0])
                self.x2Positive.append(self.independentDataSet[i][1])
            else:
                self.x1Negative.append(self.independentDataSet[i][0])
                self.x2Negative.append(self.independentDataSet[i][1])

    def arePtsStillMisclassified(self):
        for i in range(self.noOfObs):
            inputVector = self.independentDataSetWithAdditionForBiasTerm[i]
            output = self.sgn(self.dotProduct(self.weightsVector, inputVector))
            target = self.dependentValuesAsVector[i]
            if output != target:
                return True
        return False

    def applyPerceptronRule(self):
        self.weightsVector = self.initialWeightsVector
        noOfIterations = 0
        while self.arePtsStillMisclassified():
            for i in range(self.noOfObs):
                # print(" iteration-no :: ", noOfIterations)
                inputVector = self.independentDataSetWithAdditionForBiasTerm[i]
                output = self.sgn(self.dotProduct(
                    self.weightsVector, inputVector))
                target = self.dependentValuesAsVector[i]
                deltaW = self.getChangeInWeightsVector(
                    target, output, inputVector)
                self.weightsVector = self.addTwoVectors(
                    self.weightsVector, deltaW)
                # self.printWeightsTargetAndOutput()
                # self.plotDataAndHyperplaneGraph()
#                if not self.arePtsStillMisclassified() :
#                    break
                noOfIterations = noOfIterations + 1
        self.plotDataAndHyperplaneGraph()

    def printWeightsTargetAndOutput(self):
        print(self.weightsVector)
        for i in range(self.noOfObs):
            inputVector = self.independentDataSetWithAdditionForBiasTerm[i]
            output = self.sgn(self.dotProduct(self.weightsVector, inputVector))
            target = self.dependentValuesAsVector[i]
            print("Target : ", target, " | Output : ", output)

    def plotDataAndHyperplaneGraph(self):
        self.plotData()
        self.plotHyperplane()
        plt.show()

    def plotHyperplane(self):
        if self.weightsVector[2] != 0:
            intercept = -self.weightsVector[0] / self.weightsVector[2]
            slope = -self.weightsVector[1] / self.weightsVector[2]
            self.abline(slope, intercept)

    def addTwoVectors(self, u, v):
        w = []
        for i in range(len(u)):
            w.append(u[i] + v[i])
        return w

    def getChangeInWeightsVector(self, target, output, inputVector):
        changeInWeightsVector = []
        for i in range(len(inputVector)):
            y = self.dependentValuesAsVector[i]
            change = self.learningRate * (target - output) * inputVector[i]
            changeInWeightsVector.append(change)
        return changeInWeightsVector

    def dotProduct(self, x, y):
        sum = 0
        for i in range(len(x)):
            sum += x[i] * y[i]
        return sum

    def sgn(self, x):
        if x > 0:
            return 1
        else:
            return -1

    # Assumes that the first n-1 columns represent n-1 independent variables and the last column represents the dependent variable
    def readCSV(self):
        datafile = open(self.csvFilePath, "rU")
        reader = csv.reader(datafile, delimiter=",")
        self.data = []
        for row in reader:
            self.data.append(row)
        self.typeCastDataToFloat()
        self.getDataVariables()
        datafile.close()

    def typeCastDataToFloat(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.data[i][j] = float(self.data[i][j])

    def getDataVariables(self):
        self.noOfObs = len(self.data)
        # Getting rid of the dependent variable
        self.noOfIndependentVariables = len(self.data[0]) - 1
        self.independentDataSet = self.getIndependentDataSet()
        self.getIndependentDataSetWithAdditionForBiasTerm()
        self.dependentValuesAsVector = self.getDependentValuesAsVector()

    def getDependentValuesAsVector(self):
        a = []
        for entry in self.data:
            a.append(entry[-1])
        return a

    def getIndependentDataSetWithAdditionForBiasTerm(self):
        self.independentDataSetWithAdditionForBiasTerm = []
        for data in self.independentDataSet:
            self.independentDataSetWithAdditionForBiasTerm.append(data.copy())
            self.independentDataSetWithAdditionForBiasTerm[-1].insert(0, 1)

    def getIndependentDataSet(self):
        a = []
        noOfCols = len(self.data[0])
        for entry in self.data:
            a.append(entry[:noOfCols - 1])
        return a

    def initializeWeightsVector(self):
        self.initialWeightsVector = [
            0] * (self.noOfIndependentVariables + 1)  # Including Bias Weight


csvFilePath = "sample2dData.csv"
perceptronObj = perceptron(csvFilePath)
perceptronObj.start()
