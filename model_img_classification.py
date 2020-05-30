import numpy as np

def sigmoid(x):
    """
    sigmoid 함수

    Arguments:
        x:  scalar 또는 numpy array

    Return:
        s:  sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s

def relu(x):
    """
    ReLU 함수

    Arguments:
        x : scalar 또는 numpy array

    Return:
        s : relu(x)
    """
    s = np.maximum(0, x)
    
    return s

class NeuralNetwork:
    def __init__(self,layerDims, nSample):
        '''
        학습할 네트워크.

        Arguments:
            layerDims [array]: layerDims[i] 는 레이어 i의 hidden Unit의 개수 (layer0 = input layer)
            nSample: 데이터셋의 샘플 수
        '''

        self.nSample = nSample
        self.nlayer = len(layerDims)-1

        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.vel = {}
        self.s = {}
        self.cache = {}
        self.initialize_optimizer()

    def weightInit(self, layerDims):

        np.random.seed(1)
        parameters = {}

        for l in range(1, self.nlayer + 1):
            parameters['W' + str(l)] = np.random.randn(layerDims[l], layerDims[l-1]) * np.sqrt(2/layerDims[l-1])
            parameters['b' + str(l)] = np.zeros((layerDims[l], 1))
            # print("W", l,"shape",parameters['W' + str(l)].shape)
            # print("b", l,"shape",parameters['b' + str(l)].shape)

        return parameters

    # iniitialize parameter for optimizer
    def initialize_optimizer(self):

        for l in range(1, self.nlayer + 1):
            self.s['dW' + str(l)] = np.zeros_like(self.parameters['W'+str(l)])
            self.s['db' + str(l)] = np.zeros_like(self.parameters['b'+str(l)])



    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A23: network output
        '''

        ## 코딩시작
        W1, b1, W2, b2, W3, b3 = self.parameters.values()

        # print("In forward")
        # print("X shape", X.shape)
        # print("W1 shape", W1.shape, "b1 shape", b1.shape)
        # print("W2 shape", W2.shape, "b2 shape", b2.shape)
        # print("W3 shape", W3.shape, "b3 shape", b3.shape)
        Z1 = np.dot(W1, X) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = relu(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = sigmoid(Z3)

        # print("Z1 shape", Z1.shape)
        # print("Z2 shape", Z2.shape)
        # print("Z3 shape", Z3.shape)
        self.cache.update(X=X, Z1=Z1, A1=A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)

        return A3

    def backward(self,lambd=0.7):
        '''
        regularization term이 추가된 backward propagation.

        Arguments:
            lambd

        Return:
        '''

        W1, b1, W2, b2, W3, b3 = self.parameters.values()
        X, _, A1, _, A2, _, A3, Y = self.cache.values()

        dZ3 = A3 - Y
        dW3 = (1/self.nSample) * np.dot(dZ3, A2.T) + lambd/self.nSample * W3
        db3 = (1/self.nSample) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = (1/self.nSample) * np.dot(dZ2, A1.T) + lambd/self.nSample * W2
        db2 = (1/self.nSample) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = (1/self.nSample) * np.dot(dZ1, X.T) + lambd/self.nSample * W1
        db1 = (1/self.nSample) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads.update(dW1=dW1, db1=db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3)



        return


    def compute_cost(self, A3, Y, lambd=0.7):


        self.cache.update(Y=Y)
        W1, W2, W3 = self.parameters["W1"], self.parameters["W2"], self.parameters["W3"]

        ## 코딩시작

        logprobs = -(np.multiply(np.log(A3), Y) + np.multiply((1-Y),np.log(1-A3)))
        cost = 1/self.nSample * np.sum(logprobs)

        L2_reg = lambd/(2*self.nSample)*(np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        cost = cost + L2_reg

        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost

    def update_params(self, learning_rate=1.2, beta2=0.999, epsilon=1e-8):
        '''
        backpropagation을 통해 얻은 gradients를 update한다.

        Arguments:
            learning_rate:  학습할 learning rate

        Return:
        '''
        # print("init part update params")

        for l in range(1, self.nlayer + 1):
            self.s['dW' + str(l)] = self.s['dW'+str(l)] + (1-beta2) * np.power(self.grads['dW'+str(l)], 2)
            self.s['db' + str(l)] = self.s['db'+str(l)] + (1-beta2) * np.power(self.grads['db'+str(l)], 2)

            self.parameters['W'+str(l)] = self.parameters['W'+str(l)] - learning_rate * self.grads['dW'+str(l)] / np.sqrt(self.s['dW'+str(l)] + epsilon)
            self.parameters['b'+str(l)] = self.parameters['b'+str(l)] - learning_rate * self.grads['db'+str(l)] / np.sqrt(self.s['db'+str(l)] + epsilon)
            # print("sdW",l,"shape", self.s['dW' + str(l)].shape)
            # print("sdb",l,"shape", self.s['db' + str(l)].shape)
            # print("W",l,"shape", self.parameters['W'+str(l)].shape)
            # print("b",l,"shape", self.parameters['b'+str(l)].shape)


    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''

        A3 = self.forward(X)
        predictions = (A3 > 0.5)

        return predictions
