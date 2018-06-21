import math
import random
import numpy as np
np.seterr(all = 'ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dsigmoid(y):
    return y * (1.0 - y)
def tanh(x):
    return math.tanh(x)
def dtanh(y):
    return 1 - y*y

class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

        self.input = input + 1
        self.hidden = hidden
        self.output = output

        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        input_range = 1.0 / self.input ** (1/2)
        output_range = 1.0 / self.hidden ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output))

        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs')

        for i in range(self.input -1):
            self.ai[i] = inputs[i]

        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets):

        if len(targets) != self.output:
            raise ValueError('Wrong number of targets')

        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):

        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def train(self, patterns):
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            if i % 10 == 0:
                print('error %-.5f' % error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))

    def predict(self, X):

        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions

def demo():

    def load_data():
        data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')

        y = data[:,0:10]

        data = data[:,10:]
        data -= data.min()
        data /= data.max()

        out = []
        print data.shape

        for i in range(data.shape[0]):
            cat = list((data[i,:].tolist(), y[i].tolist()))
            out.append(cat)

        return out

    X = load_data()

    print X[9]

    NN = MLP_NeuralNetwork(64, 100, 10, iterations = 50, learning_rate = 0.5, momentum = 0.5, rate_decay = 0.01)

    NN.train(X)

    NN.test(X)

if __name__ == '__main__':
    demo()
