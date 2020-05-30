import numpy as np
import matplotlib.pyplot as plt
import dataUtils_3rd
import model_img_classification

import json
from json import JSONEncoder
import numpy

np.random.seed(1)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# save weights of model from json file
def saveParams(params, path):
    with open(path, "w") as make_file:
        json.dump(params, make_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")

# load weights of model from json file
def loadParams(path):
    with open(path, "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decodedArray = json.load(read_file)
    return decodedArray


def main():
    epochs = 
    learning_rate =
    batch_size =
    resume = False # path of model weights
    model_weights_path = 'weights_2020125001.json'

    ### dataset loading 하기.
    dataPath = 'dataset/train'
    valPath = 'dataset/val'
    dataloader = dataUtils.Dataloader(dataPath, minibatch=batch_size)
    val_dataloader = dataUtils.Dataloader(valPath)
    
    nSample = 
    layerDims = 

    simpleNN = model.NeuralNetwork(layerDims, nSample)
    if resume:
        simpleNN.parameters = loadParams(resume)

    for epoch in range (epochs):
        training(dataloader, simpleNN, learning_rate, epoch)
        
        if epoch%10==1:
            validation(val_dataloader,simpleNN)
            
    validation(val_dataloader,simpleNN)
    saveParams(simpleNN.parameters, model_weights_path)


def validation(dataloader, simpleNN):
    for i, (images, targets) in enumerate(dataloader):
        # do validation

def training(dataloader, simpleNN, learning_rate, epoch):

    for i, (images, targets) in enumerate(dataloader):
        # do training



    return simpleNN
 

if __name__=='__main__':
    main()