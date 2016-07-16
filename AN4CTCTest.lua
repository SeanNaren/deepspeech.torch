local Network = require 'Network'
local AN4CTCCorpus = require 'AN4CTCCorpus'
local AN4CTCEvaluate = require 'Evaluator'
local AN4PhonemeDictionary = require 'AN4PhonemeDictionary'
require 'nn'
require 'rnn'
require 'xlua'

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.t7",
    gpu = true
}

Network:init(networkParams)
print("Network loaded")

local dictionaryDirPath = "/home/sean/Work/CTCSpeechRecognition/an4.dict"
local folderDirPath = "/home/sean/Work/CTCSpeechRecognition/Audio/an4"

--Window size and stride for the spectrogram transformation.
local windowSize = 0.02 * 16000
local stride = 0.01 * 16000

AN4PhonemeDictionary.init(dictionaryDirPath)

-- Get the test set data
local testDataSet = AN4CTCCorpus.getTestingData(folderDirPath, windowSize, stride, AN4PhonemeDictionary)

-- Run the test data set through the net and print the results
local testResults = {}
local cumPER = 0
local input = torch.Tensor()
if (networkParams.gpu == true) then input = input:cuda() end

for i = 1,#testDataSet do
    local inputCPU = testDataSet[i].input
    local targets = testDataSet[i].labels
    -- transfer over to GPU
    input:resize(1,1,inputCPU:size(1),inputCPU:size(2))
    input[1][1]:copy(inputCPU)
    local prediction = Network:predict(input)

    local predictedPhones = AN4CTCEvaluate.getPredictedPhones(prediction)
    local PER = AN4CTCEvaluate.sequenceErrorRate(targets, predictedPhones)

    local targetPhoneString = ""
    local predictedPhoneString = ""

    -- Turn targets into text string
    for i = 1,#targets do
        local spacer
        if (i < #targets) then spacer = " " else spacer = "" end
        targetPhoneString = targetPhoneString .. AN4CTCCorpus.indexToPhone(targets[i]) .. spacer
    end

    -- Turn predictions into text string
    for i = 1,#predictedPhones do
        local spacer
        if (i < #predictedPhones) then spacer = " " else spacer = "" end
        predictedPhoneString = predictedPhoneString .. AN4CTCCorpus.indexToPhone(predictedPhones[i]) .. spacer
    end

    cumPER = cumPER + PER
    local row = {}
    row.PER = PER
    row.text = testDataSet[i].text
    row.predicted = predictedPhoneString
    row.target = targetPhoneString
    table.insert(testResults, row)
end

-- Print the results sorted by PER
table.sort(testResults, function (a,b) if (a.PER < b.PER) then return true else return false end end)
for i = 1,#testResults do
    local row = testResults[i]
    print(string.format("PER = %.0f%% | Text = \"%s\" | Predicted Phones = \"%s\" | Target Phones = \"%s\"",
        row.PER*100, row.text, row.predicted, row.target))
end

-- Print the overall average PER
local averagePER = cumPER / #testDataSet

print ("\n")
print(string.format("Testset Phoneme Error Rate : %.0f%%", averagePER*100))
