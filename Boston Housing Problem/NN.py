import numpy as np
import random


relu = lambda z:np.maximum(0.01 * z,z) # Relu 激活函数
relu_deriv = lambda z,a:np.where(z > 0,1.0,0.01) # Relu 激活函数的导数
sigmoid = lambda z:1/(1+np.exp(-z)) # sigmoid 激活函数
sigmoid_deriv = lambda z,a:np.multiply(a,1-a) # sigmoid 激活函数的导数

def lostFunc(A,Y): # cross entropy lost function
    assert(A.shape == Y.shape),'A.shape != Y.shape'
    # A and Y are matrix, but we just want to operation on column, so use axis = 0
    ylna = np.multiply(Y,np.log(A+1e-10)) # add 1e-10 to forbidden np.log(0)
    return -np.sum(ylna,axis = 0,keepdims=True) # remember add minus symbol "-"

def softmax(Z): # Z is a column vector, but we need to handle when Z is a matrix, use axis = 0
    maxNumber = np.max(Z,axis = 0,keepdims=True)
    assert(Z.shape[1] == maxNumber.shape[1])
    Z -= maxNumber
    Zexp = np.exp(Z)
    return Zexp / np.sum(Zexp,axis = 0,keepdims=True)

def softmax_deriv(Z,A,Y): # softmax 的导数
    assert(A.shape == Y.shape),'A.shape is not same as Y.shape'
    return A - Y

class NN:
    def __init__(self,data,layers, function,hyperParameters = {}):
        """
        data -- dict
            data['trainX']
            data['trainY']
        laysers --  iterable
                    layers[0] = n_x -- size of input
                    len(layers)-1 is the number of layers, i.e. (n_h + 1) = L
                    layers[-1] = n_y -- size of output
        function -- dict
                 -- activation: dict. Zi -> Ai
                 -- derivative: dict. dAi -> dZi
                 -- lostFunction: (AL,Y) -> cost_i
        hyperParameters -- L2 -- lambda
                            dropout
        """
        assert(isinstance(data,dict)),'data is not a dict'
        assert(isinstance(layers,list)),'layers is not a list'
        assert(isinstance(hyperParameters,dict)),'hyper parameters is not a dict'

        self.data = data
        self.caches = {'A0':data['trainX']} # save Zi = WiAi+bi and Ai, denote A0 = trainX
        self.Y = data['trainY']
        self.L = len(layers) - 1
        # Wi, bi
        self.parameters={}
        for i in range(1,self.L+1):
            self.parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) * np.sqrt(2 / layers[i-1]) # * 0.01 # 
            self.parameters['b'+str(i)] = np.zeros((layers[i],1))  
        # function
        self.function = function
        self.activation = function['activation']
        self.derivative = function['derivative']
        self.lostFunction = function['lostFunction']
        self.grads = {} # save gradients of Wi,bi in the network
        
        if 'lambda' not in hyperParameters.keys(): # set default L2 value 0.001
            hyperParameters['lambda'] = 0.001
        if 'dropout' not in hyperParameters.keys(): # set default dropout value 0.5
            hyperParameters['dropout'] = 0.5
        hyperParameters['open-dropout'] = 'open-dropout' in hyperParameters.keys() # set open-dropout True if exist, other False
        
        
        print('L2 regularition: lambda =',hyperParameters['lambda'])
        print('open-dropout:',hyperParameters['open-dropout'])
        if hyperParameters['open-dropout']:
            print('Dropout in hidden layers: keep probility is',hyperParameters['dropout'])
        if 'dropout-input' in hyperParameters.keys():
            print('dropout in input: keep probility is',hyperParameters['dropout-input'])
        self.hyperParameters = hyperParameters

    def __str__(self):
        data = repr(self.data) + '\n' + '-'*20 + '\n'
        caches = repr(self.caches) + '\n' + '-'*20 + '\n'
        parameters = repr(self.parameters) + '\n' + '-'*20 + '\n'
        grads = repr(self.grads) + '\n' + '-'*20 + '\n'
        return parameters+grads+caches

    def forwardOneLayer(self,Wi,AiPrevious,bi,activation):
        Zi = np.dot(Wi,AiPrevious) + bi
        Ai = activation(Zi)
        return Zi,Ai

    def forwardPropagation(self,parameters = None, activation = None, caches = None,L = None,hyperParameters = None):
        # L is the number of layers
        parameters = parameters or self.parameters
        activation = activation or self.activation
        caches = caches or self.caches
        L = L or self.L
        hyperParameters = hyperParameters or self.hyperParameters
        for i in range(1,L+1):
            Wi = parameters['W'+str(i)]
            AiPrevious = caches['A'+str(i-1)] # !! i-1 not i

            if i == 1 and 'dropout-input' in hyperParameters: # dropout in input
                A0 = caches['A'+str(i-1)]
                D0 = np.random.rand(*A0.shape) < hyperParameters['dropout-input']
                caches['D0'] = D0 # save D0
                AiPrevious = np.multiply(A0,D0)

            bi = parameters['b'+str(i)]
            caches['Z'+str(i)],caches['A'+str(i)] = self.forwardOneLayer(Wi,AiPrevious,bi,activation[i])
            # --- handle about dropout
            if i < L and hyperParameters['open-dropout']:
                # print('dropout now',hyperParameters['dropout'])
                Di = np.random.rand(*caches['A'+str(i)].shape) < hyperParameters['dropout']
                assert(Di.shape == caches['A'+str(i)].shape)
                caches['A'+str(i)] = np.multiply(caches['A'+str(i)],Di)  / hyperParameters['dropout']
                caches['D'+str(i)] = Di # add Di for backward propagation
                
            
    def backwardOneLayer(self,dZi,Wi,AiPrevious,compute_dAiPrevious = True): # Zi = Wi * A_pre + bi
        # cost = sum(lost{i})/dataSize , {i} mean the i-th train data
        dataSize = dZi.shape[1] 
        dWi = np.dot(dZi,AiPrevious.T)#/dataSize
        dAiPrevious = None # according compute_dAiPrevious' value to decide whether to compute dAiPrevious
        # since AiPrevious == A0, there are meanless or cost too much resource
        if compute_dAiPrevious:dAiPrevious = np.dot(Wi.T,dZi)
        dbi = np.sum(dZi,axis=1,keepdims=True)#/dataSize # so dbi{j} = sum(dZi{j})
        assert(dWi.shape == Wi.shape)
        return dWi,dAiPrevious,dbi

    def backwardPropagation(self,parameters = None,caches = None, grads = None, derivative = None, L = None, Y = None,hyperParameters = None):
        if parameters is None: parameters = self.parameters
        if caches is None: caches = self.caches
        if grads is None: grads = self.grads
        if derivative is None: derivative = self.derivative
        if L is None: L = self.L
        if Y is None: Y = self.Y
        if hyperParameters is None: hyperParameters = self.hyperParameters
        dataSize = Y.shape[1]
        # compute dZL
        ZL,AL,Y = caches['Z'+str(L)],caches['A'+str(L)],Y
        dZL = 1/dataSize * derivative[L](ZL,AL,Y) # watch here
        grads['dZ'+str(L)] = dZL # add dZ{L} to gradient
        for i in reversed(range(1,L+1)):
            dZi = grads['dZ'+str(i)]
            Wi, AiPrevious, bi = parameters['W'+str(i)], caches['A'+str(i-1)], parameters['b'+str(i)]
            assert(dZi.shape[0] == bi.shape[0]),'dZi.shape[0] != bi.shape[0]'
            dWi,dAiPrevious,dbi = self.backwardOneLayer(dZi,Wi,AiPrevious,i>1) # compute gradient of Wi,bi
            # add gradient in self.grads
            grads['dW'+str(i)] = dWi
            grads['db'+str(i)] = dbi
            if i > 1 : # since there are no dA0 and dZ0
                grads['dA'+str(i-1)] = dAiPrevious
                # print('dA'+str(i-1),dAiPrevious)
                grads['dZ'+str(i-1)] = dAiPrevious * derivative[i-1](caches['Z'+str(i-1)],AiPrevious) # don't forget multiple dAiPrevious
                # print('dZ'+str(i-1),self.grads['dZ'+str(i-1)])
                # --- handle dropout about
                if hyperParameters['open-dropout']:
                    grads['dA'+str(i-1)] = np.multiply(grads['dA'+str(i-1)],caches['D'+str(i-1)]) / hyperParameters['dropout'] # whatever, it not necessry here
                    grads['dZ'+str(i-1)] = np.multiply(grads['dZ'+str(i-1)],caches['D'+str(i-1)]) / hyperParameters['dropout'] # !!!!! dZ[i-1] must turn off neorual
            
    def updateParameters(self,learningRate,parameters = None,grads = None):
        # lambd is L2 regularization
        lambd = self.hyperParameters['lambda']
        if parameters is None: parameters = self.parameters
        if grads is None: grads = self.grads
        dataSize = self.caches['A0'].shape[1] # get dataSize for L2 regularization
        for i in range(1,self.L+1):
            parameters["W" + str(i)] -= learningRate * grads['dW'+str(i)] + (lambd / dataSize) * parameters["W" + str(i)]
            parameters["b" + str(i)] -= learningRate * grads['db'+str(i)]

    def computeCost(self,AL=None,Y=None,lostFunction=None):
        """
        lostFunction : (A[L]{i},Y{i}) -> cost{i}
        """
        if AL is None:AL = self.caches['A'+str(self.L)]
        if Y  is None:Y = self.Y
        if lostFunction is None:lostFunction=self.lostFunction 
        dataSize = Y.shape[1]
        assert(AL.shape == Y.shape),"AL.shape != Y.shape"
        losts = lostFunction(AL,Y)
        cost = np.sum(losts) / dataSize
        assert(cost.shape == ())
        return cost

    def computeCostWithL2(self,AL=None,Y=None,lostFunction=None):
        """
        lostFunction : (A[L]{i},Y{i}) -> cost{i}
        """
        cost = self.computeCost(AL,Y,lostFunction)
        Y = Y or self.Y
        m = Y.shape[1] # number of data
        lambd = self.hyperParameters['lambda']
        L2cost = 0
        for key in self.parameters:
            if key[0] == 'W':
                L2cost += np.sum( lambd / 2 / m * np.power(self.parameters[key],2) )
        # print('L2cost:',L2cost)
        return cost + L2cost

    def predict(self,parameters = None,X = None,activation = None,predictFunction = None):# it's not general for predict
        # compute A[L]
        A = X # input data
        # is optional parameter is None, use class's function build in
        if parameters is None: parameters = self.parameters
        if X is None: A = self.caches['A0']
        if activation is None: activation = self.activation
        if predictFunction is None: predictFunction = self.function['predictFunction']
        # compute the last layer, i.e A[L]
        for i in range(1,self.L+1):
            Wi,bi = parameters['W'+str(i)],parameters['b'+str(i)]
            Zi = np.dot(Wi,A) + bi
            A = activation[i](Zi)
        return predictFunction(A)

    def accuracy(self,predictions,labels,accuracyFunction = None):
        assert(predictions.shape == labels.shape),'prediction.shape != labels.shape'
        if accuracyFunction is None: accuracyFunction = self.function['accuracyFunction']
        return accuracyFunction(predictions,labels) # take 2 parameters

    def oneBatch(self,learningRate,X = None, Y = None):
        # X is train data
        # Y is train label
        assert(type(learningRate) == float),'learningRate is not a float'
        if X is None: X = self.caches['A0']
        if Y is None: Y = self.Y
        self.caches['A0'] = X
        self.Y = Y
        self.forwardPropagation()
        self.backwardPropagation()
        self.updateParameters(learningRate)
    
    def miniBatch(self,learningRate, batchSize, getCost = False):
        # epoch with mini batch
        dataSize = self.data['trainY'].shape[1]
        # print('dataSize:',dataSize)
        permutation = np.random.permutation(dataSize)
        shuffledX = self.data['trainX'][:,permutation]
        shuffledY = self.data['trainY'][:,permutation]
        # if dataSize = 9, batchSize = 4, since 4 + 4 + 1 = 9, so first train 4, next train 4, no train 1
        costs = []
        batchI = 0
        while batchI * batchSize + batchSize <= dataSize:
            miniX = shuffledX[:,batchI * batchSize:batchI * batchSize + batchSize]
            miniY = shuffledY[:,batchI * batchSize:batchI * batchSize + batchSize]
            # print('data range:',batchI * batchSize,'--',batchI * batchSize + batchSize)
            self.oneBatch(learningRate,X = miniX,Y = miniY) # with side effect!!
            batchI += 1 # update batch index
            if getCost: costs.append(self.computeCost())
        return costs

    def miniBatchRandom(self,learningRate, batchSize, batchTimes, getCost = False):
        # batchTimes -- use batchSize's data to train batchTimes times
        dataSize = self.data['trainY'].shape[1]
        # print('dataSize:',dataSize)
        costs = []
        for i in range(batchTimes): # every time choose batchSize's data form train data randomly
            permutation = np.random.permutation(dataSize)[:batchSize]
            # print('permutation.shape:',permutation.shape)
            # print(permutation)
            miniX = self.data['trainX'][:,permutation]
            miniY = self.data['trainY'][:,permutation]
            self.oneBatch(learningRate = learningRate,X = miniX, Y = miniY)
            if getCost:
                costs.append(self.computeCostWithL2()) # watch out here, the function to compute cost
        return costs

    def train(self,learningRate,trainTimes,printCostTimes = None):
        assert(type(learningRate)==float),'learningRate is not float'
        assert(type(trainTimes)==int),'train times is not int'
        costs = [] # save every printCostTimes
        for i in range(trainTimes):
            self.oneBatch(learningRate) # use one batch to train one time
            if type(printCostTimes)==int and i % printCostTimes == 0:
                cost = self.computeCost()
                costs.append(cost)
                print('cost after',str(i),'iteration:',cost)
        return costs

if __name__ == 'main':
    pass
else:
    print('my Neural Network import succeed')