--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain.]]

local Network = require 'Network'

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = arg[1] or "./models/CTCNetwork.t7", -- Rename the evaluated model to CTCNetwork.t7 or pass the file path as an argument.
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 1, -- Number of GPUs, set -1 to use CPU
    trainingSetLMDBPath = './prepare_an4/train/', -- online loading path
    validationSetLMDBPath = './prepare_an4/test/',
    logsTrainPath = './logs/TrainingLoss/',
    logsValidationPath = './logs/TestScores/',
    modelTrainingPath = './models/',
    dictionaryPath = './an4.phone',
    batchSize = 1,
    validationBatchSize = 1,
    validationIterations = 130 -- batch size 1, goes through 130 samples.
}

Network:init(networkParams)

print("Testing network...")
local wer = Network:testNetwork()
print(string.format('Number of iterations: %d average WER: %2.f%%', networkParams.validationIterations, 100 * wer))
print(string.format('More information written to log file at %s', networkParams.logsValidationPath))
