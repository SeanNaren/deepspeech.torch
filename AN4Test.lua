local AudioData = require 'AudioData'
local Network = require 'Network'

--Finds the index of the max value in the 1d table given. Used when retrieving the prediction from the network.
function maxIndex(table)
    local maxIndex = 1
    local maxValue = 0
    for index, value in ipairs(table) do
        if (value > maxValue) then maxValue = value maxIndex = index end
    end
    return maxIndex
end

--The mini-batch size.
local batchSize = 5

--Training parameters
local epochs = 50
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 10e-4,
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

--Convert the dataset into a padded dataset of all same sizes.
local dataset = Network.createDataSet(testInputs, testTargets, batchSize)

--For testing purposes, we predict the first test data in the dataset and retrieve the first prediction.
--TODO this should be an accuracy checker where we loop through all samples to retrieve an accuracy value.
local sample1 = torch.totable(Network.predict(net, dataset:nextData())[1])

--iterate through the results of the prediction and output the letter that was predicted in the sample.
for index, result in ipairs(sample1) do
    --If the index is 1, that means that the character was blank.
    if (maxIndex(result) ~= 1) then print(AudioData.findLetter((maxIndex(result)))) else print("blank") end
end

print("finished")

--Creates the loss plot.
--TODO when high mem usage, this function causes a crash related to gnuplot running out of memory.
Network.createLossGraph()

--prevents script from ending to view loss graph.
while (true) do end