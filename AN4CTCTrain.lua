--[[Trains the CTC model using the AN4 audio database. Training time as of now takes less than 40 minutes on a GTX 970.]]

local Network = require 'Network'

--Training parameters
torch.setdefaulttensortype('torch.FloatTensor')
seed = 10
torch.manualSeed(seed)
cutorch.manualSeedAll(seed)
local epochs = 100

local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork.t7",
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 2, -- Number of GPUs, set -1 to use CPU
--    lmdb_path = './prepare_librispeech/train/',-- online loading path
--    val_path = './prepare_librispeech/test/',
    lmdb_path = './prepare_an4/train/',
    val_path = './prepare_an4/test/',
    dict_path = './dictionary',
    batch_size = 40,
    test_batch_size = 2,
    test_iter = 65,
    snap_shot_epochs = 20
}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 1e-3,
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
