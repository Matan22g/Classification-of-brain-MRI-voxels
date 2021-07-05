"""By Matan Achiel 205642119, Netanel Moyal 307888974"""

import numpy as np


class FC_NN(object):
    def __init__(self, layers=[1024, 1024, 1], activations=['sigmoid', 'sigmoid'], step_lr=0, min_delta_loss=1e-3):
        assert (len(layers) == len(activations) + 1)
        self.layers = layers
        self.activations = activations
        self.step_lr = step_lr
        self.min_delta_loss = min_delta_loss
        self.reset_weights()

    def reset_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            self.weights.append(0.5 * np.random.randn(self.layers[i], self.layers[i + 1]))
            self.biases.append(0.5 * np.random.randn(1, self.layers[i + 1]))

    def fix_weights(self):
        for i in range(len(self.layers) - 1):
            self.weights[i] = self.weights[i].T
            self.biases[i] = self.biases[i].T

    def feedforward(self, x):
        # return the feedforward value for x
        a = np.copy(x).T
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return z_s, a_s

    def backpropagation(self, y, z_s, a_s):
        # error for each layer
        deltas = [None] * len(self.weights)

        # last layer error
        deltas[-1] = ((y - a_s[-1]) * (self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))

        # BackPropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * (
                self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))

        batch_size = len(y)
        db = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T) / float(batch_size) for i, d in enumerate(deltas)]

        # derivatives respect to weight matrix and biases
        return dw, db

    def shuffle_train(self, x, y):
        N = len(x)
        batch_ind_lst = np.random.choice(N, N, replace=False)
        new_x = []
        new_y = []
        for i in batch_ind_lst:
            new_x.append(x[i])
            new_y.append(y[i])
        return new_x, new_y

    def train(self, x, y, test, batch_size=10, epochs=100, lr=0.01, epoch_checker=100):
        self.fix_weights()

        iter = 0
        total_loss = []
        epoch_avg_loss = []
        acc_per_epoch = []
        e = 0
        check_first_acc = True
        check_acc_epoch = epoch_checker
        while e < epochs:
            i = 0
            x, y = self.shuffle_train(x, y)
            while i < len(y):
                iter += 1
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                i = i + batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [w + lr * dweight for w, dweight in zip(self.weights, dw)]
                self.biases = [w + lr * dbias for w, dbias in zip(self.biases, db)]
                loss = np.linalg.norm(a_s[-1] - y_batch)
                total_loss.append(loss)

            """Compute Avg loss, and Acc"""
            curr_avg_loss = np.mean(total_loss)
            acc = self.test(test)
            acc_per_epoch.append(acc)
            epoch_avg_loss.append(curr_avg_loss)

            print(f'\rEpoch: {e}, Accuracy (On Test) = {round(acc, 2)}%, Avg Loss = {round(curr_avg_loss, 2)}', end='',
                  flush=True)

            e += 1
            if e == epochs:
                inc_amount = 50
                epochs += inc_amount

            """Stopping if acc has'nt change over 30 epochs"""
            if len(acc_per_epoch) > check_acc_epoch:
                if round(acc_per_epoch[-1], 4) - round(acc_per_epoch[-check_acc_epoch], 4) == 0:
                    print("\nAccuracy hasn't change for while, stopping...\n")
                    break

            """reset_weights if acc has'nt change from the first epoch"""
            if check_first_acc:
                if len(acc_per_epoch) > 5:
                    if round(acc_per_epoch[-1], 4) <= 55:
                        self.reset_weights()
                        self.fix_weights()
                    else:
                        check_first_acc = False
        self.fix_weights()

        return epoch_avg_loss, acc_per_epoch, e

    @staticmethod
    def getActivationFunction(name):
        if name == 'sigmoid':
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif name == 'linear':
            return lambda x: x
        elif name == 'relu':
            def relu(x):
                y = np.copy(x)
                y[y < 0] = 0
                return y

            return relu
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x

    @staticmethod
    def getDerivitiveActivationFunction(name):
        if name == 'sigmoid':
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'linear':
            return lambda x: 1
        elif name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y

            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1

    def test(self, test):
        z_s, a_s = self.feedforward(test[0])
        predictions = np.round(a_s[-1])
        diff = predictions - test[1]
        right_amount = np.count_nonzero(diff == 0)
        return 100 * (right_amount / len(test[1]))
