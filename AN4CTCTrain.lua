local AudioData = require 'AudioData'
local Network = require 'Network'

--[[Takes the resulting predictions from the network and prints the letter equivalent in one string.]]
local function printPredictions(predictions, testSample)
    local string = ""
    local prevLetter = ""
    -- Iterate through the results of the prediction and append the letter that was predicted in the sample.
    for index, prediction in ipairs(predictions) do
        local maxValue, maxIndex = torch.max(prediction, 1)
        -- We minus 1 to the index because a CTC blank has the index of 0.
        maxIndex = maxIndex[1] - 1
        -- If the index is 0, that means that the character was blank.
        if (maxIndex ~= 0) then
            local letter = AudioData.findLetter(maxIndex)
            if (letter ~= prevLetter) then
                string = string .. letter
                prevLetter = letter
            end
        end
    end
    local targetSentence = ""
    for index, characterIndex in ipairs(testSample) do
        targetSentence = targetSentence .. AudioData.findLetter(characterIndex)
    end
    print("Actual target: " .. targetSentence)
    print("Prediction: " .. string)
end

--Training parameters
local epochs = 15

local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork"
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

--The test set in spectrogram tensor form.
local testDataSet = AudioData.retrieveAN4TestDataSet(an4FolderDir, windowSize, stride)

--Creates the loss plot.
Network:createLossGraph()

--For testing purposes, we predict certain items from the dataset.
--TODO should use a language model or Seq2Seq to handle spell checking.
--TODO this should be an accuracy checker where we loop through all samples to retrieve an accuracy value.
local numberOfTestSamples = 5
for i = 0, numberOfTestSamples do
    local inputs, targets = testDataSet:nextData()
    local predictions = Network:predict(inputs)
    printPredictions(predictions, targets[1])
end

print("finished")

--prevents script from ending to view loss graph.
while (true) do end