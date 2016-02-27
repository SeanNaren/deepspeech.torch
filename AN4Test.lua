local AudioData = require 'AudioData'
local Network = require 'Network'

--Finds the index of the max value in the 1d table given. Used when retrieving the prediction from the network.
local function maxIndex(table)
    local maxIndex = 1
    local maxValue = 0
    for index, value in ipairs(table) do
        if (value > maxValue) then maxValue = value maxIndex = index end
    end
    --We -1 from the index since CTC starts at index 0.
    return maxIndex - 1
end

--Takes the resulting predictions from the network and prints the letter equivalent in one string.
local function printPredictions(predictions)
    local string = ""
    --iterate through the results of the prediction and append the letter that was predicted in the sample.
    for index, result in ipairs(predictions) do
        --If the index is 0, that means that the character was blank.
        if (maxIndex(result) ~= 0) then
            string = string .. " " .. (AudioData.findLetter((maxIndex(result))))
        end
    end
    print(string)
end

--The mini-batch size.
local batchSize = 1

--Training parameters
local epochs = 1000
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 0.001,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9
}

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 128

--The training set in spectrogram tensor form.
local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"
local inputs, targets = AudioData.retrieveAN4TrainingDataSet(an4FolderDir, windowSize, stride)

--Create and train the network based on the parameters and training data.
local net = Network.createSpeechNetwork()
Network.trainNetwork(net, inputs, targets, batchSize, epochs, sgdParams)

--The test set in spectrogram tensor form.
local testInputs, testTargets = AudioData.retrieveAN4TestDataSet(an4FolderDir, windowSize, stride)

--TODO currently this dataset is AN EXACT REPLICA OF THE ABOVE, and does not use testInputs. This is to see
--TODO if the network can correctly align the first sample.
--Convert the dataset into a padded dataset of all same sizes.
local dataset = Network.createDataSet(inputs, targets, batchSize)

--For testing purposes, we predict certain items from the dataset.
--TODO this should be an accuracy checker where we loop through all samples to retrieve an accuracy value.
printPredictions(torch.totable(Network.predict(net, dataset:nextData())[1]))
printPredictions(torch.totable(Network.predict(net, dataset:nextData())[1]))
printPredictions(torch.totable(Network.predict(net, dataset:nextData())[1]))
printPredictions(torch.totable(Network.predict(net, dataset:nextData())[1]))
printPredictions(torch.totable(Network.predict(net, dataset:nextData())[1]))


print("finished")

--Creates the loss plot.
--TODO when high mem usage, this function causes a crash related to gnuplot running out of memory.
Network.createLossGraph()

--prevents script from ending to view loss graph.
while (true) do end