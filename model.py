import numpy as np
from math import exp


def tansig(x, x0):
    return (1-exp(-x/x0)) / (1+exp(-x/x0))

def d_tansig(x, x0):
    return (1/(2*x0))*(1+tansig(x, x0))*(1-tansig(x, x0))

# N0 - N1 - N2
# transfer function transig - x0
# input neurons are bipolar
# N2 = 1?
# Tolerance T = 0.05
# cost function = quadratic (L2)/ bipolar cross entropy (BCE)
# initialization - zeta
# epoch_thres = I
class Model:
    def __init__(self, N0, N1, N2, x0, lr, tol, zeta=1, 
                 epoch_thres=1000, cost='L2', transfer='tansig'):
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2
        self.x0 = x0
        self.lr = lr
        self.tol = tol
        self.zeta = zeta
        if (cost != 'L2') and (cost != 'BCE'):
            raise ValueError("Unknown input cost function.")
        self.cost = cost
        self.epoch_thres = epoch_thres
        self.transfer = transfer

        self.__build_transfer()
        messages = ["Model setup:", "Network: {}-{}-{}", \
                    "x0: {}; learning rate: {}; Tolerance: {};", \
                    "zeta: {}; maximum running epochs: {};", \
                    "Cost function: {}; transfer function: {}"]
        print("\n".join(messages).format(N0, N1, N2, x0, lr, tol, zeta, \
               epoch_thres, cost, transfer))


    def __build_weights(self, input_weights=False, w01=None, w12=None):
        if input_weights:
            # check input weights' dimensions
            assert w01.shape[0]==(self.N0+1), "Dim 0 of w01 incorrect, \
                expect {} but receive {}".format(self.N0+1, w01.shape[0])
            assert w01.shape[1]==(self.N1), "Dim 1 of w01 incorrect, \
                expect {} but receive {}".format(self.N1, w01.shape[1])
            assert w12.shape[0]==(self.N1+1), "Dim 0 of w12 incorrect, \
                expect {} but receive {}".format(self.N1+1, w12.shape[0])
            assert w12.shape[1]==(self.N2), "Dim 1 of w12 incorrect, \
                expect {} but receive {}".format(self.N2, w12.shape[1])
            
            self.w01 = w01
            self.w12 = w12

        else:
            self.w01 = np.random.rand(self.N0+1, self.N1) * \
                        (2 * self.zeta) - self.zeta
            self.w12 = np.random.rand(self.N1+1, self.N2) *\
                        (2 * self.zeta) - self.zeta

    def __build_transfer(self):
        if self.transfer != 'tansig':
            raise ValueError("Unknown transfer function")
        self.vfunc_tran = np.vectorize(lambda x: tansig(x, self.x0))
        self.vfunc_dtran = np.vectorize(lambda x: d_tansig(x, self.x0))

    # X: input
    # Y: target
    # epochs: number of epochs to run; if not specified then -1
    def train(self, X, Y, epochs=-1, input_weights=False, w01=None, w12=None):
        # initialize the weights
        self.__build_weights(input_weights, w01, w12)

        converge = False
        counter = 0
        j = 0
        num_sample = X.shape[0]
        if epochs > 0:
            thres = min(epochs, self.epoch_thres)
        else:
            thres = self.epoch_thres

        while (not converge) and (j < thres):

            square_error = 0.
            for i in range(num_sample):

                # append 1 at front for bias, dim = (N0+1, )
                a0 = np.concatenate([[1], X[i]], axis=0)
                # n1: dim = (N1, )
                n1 = np.dot(a0, self.w01)
                a1 = self.vfunc_tran(n1)
                # a1: dim = (N1+1, )
                a1 = np.concatenate([[1], a1], axis=0)
                # n2: dim = (N2, )
                n2 = np.dot(a1, self.w12)
                a2 = self.vfunc_tran(n2)

                square_error += 0.5 * ((a2 - Y[i])**2).sum()

                # s2: dim = (N2, )
                # if cost='BCE': s2 = (a2-y)
                # if cost='L2': s2 = (a2-y)*d_tran(n2)
                s2 = (a2 - Y[i]) 
                if self.cost == 'L2':
                    s2 = s2 * self.vfunc_dtran(n2)
                # s1: dim = (N1, )
                s1 = self.vfunc_dtran(n1)*np.dot(self.w12[1:,:], s2)
                # (N1+1, N2) = (N1+1, 1) * (1, N2)
                delta_12 = a1.reshape(-1,1) * s2.reshape(1,-1)
                # (N0+1, N1) = (N0+1, 1) * (1, N2)
                delta_01 = a0.reshape(-1,1) * s1.reshape(1,-1)

                self.w12 = self.w12 - self.lr * delta_12
                self.w01 = self.w01 - self.lr * delta_01

            converge = (square_error<self.tol)
            j += 1
            print("Epoch: {0}; square error: {1:.4f}".format(j, square_error))
        self.stop_epoch = j


    def predict(self, X):
        a0 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=-1)
        a1 = self.vfunc_tran(np.dot(a0, self.w01))
        
        a1 = np.concatenate([np.ones((a1.shape[0], 1), dtype=np.float32), a1], axis=-1)
        a2 = self.vfunc_tran(np.dot(a1, self.w12))
        return a2
