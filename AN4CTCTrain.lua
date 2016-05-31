--[[Trains the CTC model using the AN4 audio database.]]

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
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 2, -- Number of GPUs, set -1 to use CPU
    trainingSetLMDBPath = './prepare_an4/train/',-- online loading path data.
    validationSetLMDBPath = './prepare_an4/test/',
    logsTrainPath = './logs/TrainingLoss/',
    logsValidationPath = './logs/ValidationScores/',
    modelTrainingPath = './models/',
    fileName = 'CTCNetwork.t7',
    dictionaryPath = './dictionary',
    batchSize = 20,
    validationBatchSize = 2,
    validationIterations = 65,
    saveModelIterations = 50
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

Network:trainNetwork(epochs, sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")
