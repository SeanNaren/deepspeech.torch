local AudioData = require 'AudioData'
local Network = require 'Network'
local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"
local inputs,targets = AudioData.retrieveAN4TrainingDataSet(an4FolderDir)
local net = Network.createAn4SmallNetwork()
local batchSize = 100
local epochs = 50
local inputs,targets = AudioData.retrieveAN4TestDataSet(an4FolderDir)
Network.trainNetwork(net, inputs, targets, batchSize, epochs)

print(Network.predict(net,inputs))