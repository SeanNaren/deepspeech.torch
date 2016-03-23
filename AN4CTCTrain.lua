local AudioData = require 'AudioData'
local Network = require 'Network'

--Training parameters
local epochs = 47

local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork.model"
}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 0.001,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 128

--The training set in spectrogram tensor form.
local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"
local trainingDataSet = AudioData.retrieveAN4TrainingDataSet(an4FolderDir, windowSize, stride)

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(trainingDataSet, epochs, sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")