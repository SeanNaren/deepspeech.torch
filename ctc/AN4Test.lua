local AudioData = require 'AudioData'
local Network = require 'Network'
local inputs,targets = AudioData.retrieveAN4DataSet("/root/IdeaProjects/CTCNetworkClient/Audio/an4")
local net = Network.createAn4SmallNetwork()
local batchSize = 20
Network.trainNetwork(net, inputs, targets, batchSize)