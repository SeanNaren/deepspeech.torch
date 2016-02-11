local AudioData = require 'AudioData'
local Network = require 'Network'

function maxIndex(table)
    local maxIndex = 1
    local maxValue = 0
    for index, value in ipairs(table) do
        if (value > maxValue) then maxValue = value maxIndex = index end
    end
    return maxIndex
end

local batchSize = 20
local epochs = 10
local windowSize = 256
local stride = 128

local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"
local inputs, targets = AudioData.retrieveAN4TrainingDataSet(an4FolderDir, windowSize, stride)

local net = Network.createTempSpeechNetwork()
Network.trainNetwork(net, inputs, targets, batchSize, epochs)

local testInputs, testTargets = AudioData.retrieveAN4TestDataSet(an4FolderDir, windowSize, stride)

local dataset = Network.createDataSet(testInputs, testTargets, batchSize)
local sample1 = torch.totable(Network.predict(net,dataset:nextData())[1])

for index, result in ipairs(sample1) do
    if (maxIndex(result) ~= 1) then print(AudioData.findLetter((maxIndex(result)))) else print("blank") end
end

print("finished")

Network.createLossGraph()

--prevents script from ending to view loss graph.
while (true) do end