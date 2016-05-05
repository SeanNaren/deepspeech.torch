--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain and a simple spell checker.]]

local Network = require 'Network'
local AN4CTCCorpus = require 'AN4CTCCorpus'
local Evaluator = require 'Evaluator'
require 'nn'
require 'rnn'

gpu = true -- Set to true if you trained a GPU based model.

function fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 75

local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.t7",
    gpu = true
}

Network:init(networkParams)
print("Network loaded")

-- Get the test set data
local testDataSet = AN4CTCCorpus.getTestingData(an4FolderDir, windowSize, stride)

-- Run the test data set through the net and print the results
local testResults = {}
local cumWER = 0
local input = torch.Tensor()
if (networkParams.gpu == true) then input = input:cuda() end

for i = 1,#testDataSet do
    local inputCPU = testDataSet[i].input
    local targets = testDataSet[i].label
    -- transfer over to GPU
    input:resize(1,1,inputCPU:size(1),inputCPU:size(2))
    input[1][1]:copy(inputCPU)
    local prediction = Network:predict(input)

    local predictedPhones = Evaluator.getPredictedCharacters(prediction)
    local WER = Evaluator.sequenceErrorRate(targets, predictedPhones)

    local targetPhoneString = ""
    local predictedPhoneString = ""

    -- Turn targets into text string
    for i = 1,#targets do
        local spacer
        if (i < #targets) then spacer = " " else spacer = "" end
        targetPhoneString = targetPhoneString .. AN4CTCCorpus.findLetter(targets[i]) .. spacer
    end

    -- Turn predictions into text string
    for i = 1,#predictedPhones do
        local spacer
        if (i < #predictedPhones) then spacer = " " else spacer = "" end
        predictedPhoneString = predictedPhoneString .. AN4CTCCorpus.findLetter(predictedPhones[i]) .. spacer
    end

    cumWER = cumWER + WER
    local row = {}
    row.WER = WER
    row.text = testDataSet[i].text
    row.predicted = predictedPhoneString
    row.target = targetPhoneString
    table.insert(testResults, row)
end

-- Print the results sorted by WER
table.sort(testResults, function (a,b) if (a.WER < b.WER) then return true else return false end end)
for i = 1,#testResults do
    local row = testResults[i]
    print(string.format("WER = %.0f%% | Text = \"%s\" | Predicted characters = \"%s\" | Target characters = \"%s\"",
        row.WER*100, row.text, row.predicted, row.target))
end
print("-----------------------------------------")
print("Individual WER above are from low to high")

-- Print the overall average PER
local averageWER = cumWER / #testDataSet

print ("\n")
print(string.format("Testset Word Error Rate : %.0f%%", averageWER*100))