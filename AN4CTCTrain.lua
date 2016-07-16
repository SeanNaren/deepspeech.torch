local AN4CTCCorpus = require 'AN4CTCCorpus'
local AN4CTCBatcher = require 'Batcher'
local Network = require 'Network'
local DeepSpeechModel = require 'DeepSpeechModel' -- for cpu mode switch to DeepSpeechModelCPU
local AN4PhonemeDictionary = require 'AN4PhonemeDictionary'

--Training parameters
local epochs = 70

local GRU = false
local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork.t7",
    model = DeepSpeechModel(GRU),
    gpu = true
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

local dictionaryDirPath = "/home/sean/Work/CTCSpeechRecognition/an4.dict"
local folderDirPath = "/home/sean/Work/CTCSpeechRecognition/Audio/an4"

AN4PhonemeDictionary.init(dictionaryDirPath)

--Window size and stride for the spectrogram transformation.
local windowSize = 0.02 * 16000
local stride = 0.01 * 16000

--The larger this value, the larger the batches, however the more padding is added to make variable sentences the same.
local maxSizeDiff = 0 -- Setting this to zero makes it batch together the same length sentences.

--The training set in spectrogram tensor form.
local inputsAndTargets = AN4CTCCorpus.getTrainingData(folderDirPath, windowSize, stride, AN4PhonemeDictionary)
local trainingBatch = AN4CTCBatcher.createMinibatchDataset(inputsAndTargets, maxSizeDiff)

-- The validation dataset used to monitor training
local testDataSet = AN4CTCCorpus.getTestingData(folderDirPath, windowSize, stride, AN4PhonemeDictionary)

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(trainingBatch, epochs, sgdParams, testDataSet)

--Creates the loss plot.
Network:createLossGraph()

print("finished")