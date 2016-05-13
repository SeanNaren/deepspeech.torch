--[[Trains the CTC model using the AN4 audio database. Training time as of now takes less than 40 minutes on a GTX 970.]]

local Network = require 'Network'

--Training parameters
torch.setdefaulttensortype('torch.FloatTensor')
seed = 10
torch.manualSeed(seed)
cutorch.manualSeedAll(seed)
local epochs = 70

local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork.t7",
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 1, -- Number of GPUs, set -1 to use CPU
    lmdb_path = 'prepare_an4/train/',-- online loading path
    val_path = 'prepare_an4/test/',
    dict_path = './dictionary',
    batch_size = 20,
    test_batch_size = 1,
    test_iter = 10
}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 1e-4,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

-- Network:trainNetwork(trainingDataSet, nil, epochs, sgdParams)
-- lets test jit loading
Network:trainNetwork(epochs, sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")
