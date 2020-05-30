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
    s = np.maximum(0,x)
    
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



        return parameters

    # iniitialize parameter for optimizer
    def initialize_optimizer(self):

        for l in range(1,self.nlayer+1):
            


    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A23: network output
        '''

        ## 코딩시작 




        return A3

    def backward(self,lambd=0.7):
        '''
        regularization term이 추가된 backward propagation.

        Arguments:
            lambd

        Return:
        '''






        return


    def compute_cost(self, A3, Y, lambd=0.7):
 




        
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


        return 

    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''

        A3=
        predictions=

        return predictions
