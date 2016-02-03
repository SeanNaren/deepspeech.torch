local AudioData = require 'AudioData'
local Network = require 'Network'

local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"
local inputs,targets = AudioData.retrieveAN4TrainingDataSet(an4FolderDir)

local net = Network.createSpeechNetwork()
local batchSize = 5
local epochs = 500
Network.trainNetwork(net, inputs, targets, batchSize, epochs)

local inputs,targets = AudioData.retrieveAN4TestDataSet(an4FolderDir)
print(Network.predict(net, {inputs[1]}))