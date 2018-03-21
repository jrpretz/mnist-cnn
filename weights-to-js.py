import sys
import h5py
import json
import numpy as np
import keras

model = keras.models.load_model("trained-model-keras.hd5")

params = {}

# for this model, there are 2 weights for each layer with weights
weightNames = []
for l in model.layers:
    if(len(l.weights) > 0):
        weightNames.append("W" + l.name)
        weightNames.append("b" + l.name)

weights = model.get_weights()

for i in range(len(weights)):
    params[weightNames[i]] = weights[i].tolist()
    
print("var getMNISTCNNParams = function(){ return %s}"%(json.dumps(params)))
