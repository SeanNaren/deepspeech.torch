module(...,package.seeall)
require 'nn'
network = require('Network')
local NetworkHandler = {}
local function extractInputsAndLabels(json)
    local inputsTensors = {}
    local labels = {}
    for trainingIndex, trainingSample in pairs(json["training"]) do
        local inputs = {}
        local targets = {}
        for inputIndex, inputSample in pairs(trainingSample["input"]) do
            table.insert(inputs,inputSample)
        end
        for outputIndex, outputSample in pairs(trainingSample["output"]) do
            table.insert(targets,outputSample)
        end
        table.insert(inputsTensors,torch.Tensor(inputs))
        table.insert(labels,targets)
    end
    return inputsTensors, labels
end

local function formatInputTensors(json)
    local inputs = {}
    for trainingIndex, trainingSample in pairs(json["prediction"]) do
            table.insert(inputs,trainingSample)
    end
    local tensorInput = torch.Tensor(inputs)
    return tensorInput
end

function NetworkHandler.trainNetwork(json)
    local inputs, labels = extractInputsAndLabels(json)
    local net = network.createNewNetwork()
    network.trainNetwork(net, inputs, labels)
    return net
end

function NetworkHandler.predictNetwork(net,json)
    local inputTensors = formatInputTensors(json)
    local prediction = network.predict(net,inputTensors)
    return prediction
end

return NetworkHandler