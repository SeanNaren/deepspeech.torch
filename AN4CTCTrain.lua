--[[Trains the CTC model using the AN4 audio database. Training time as of now takes less than 40 minutes on a GTX 970.]]

local AudioData = require 'AudioData'
local Network = require 'Network' -- To use the CPU instead change the require to NetworkCPU.
local Batcher = require 'Batcher'

local deepSpeechModel = require 'DeepSpeechModel' -- The script that contains the model we will be training.

--Training parameters
local epochs = 70

local GRU = false -- When set to true we convert all LSTMs to GRUs.
local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork.model",
    model = deepSpeechModel(GRU)
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

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 75

--The larger this value, the larger the batches, however the more padding is added to make variable sentences the same.
local maximumSizeDifference = 0 -- Setting this to zero makes it batch together the same length sentences.

--The training set in spectrogram tensor form.
local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"
local inputsAndTargets = AudioData.retrieveAN4TrainingDataSet(an4FolderDir, windowSize, stride)
local trainingDataSet = Batcher.createMinibatchDataset(inputsAndTargets, maximumSizeDifference)

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(trainingDataSet, epochs, sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")