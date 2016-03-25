--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain and a simple spell checker.]]

local SpellingChecker = require 'SpellingChecker'
local Network = require 'Network'
local AudioData = require 'AudioData'
require 'nn'
require 'xlua'

--[[Takes the resulting predictions and the transcript sentence. Returns tables of words said in both.]]
local function getWords(predictions, targetSentence)
    local predictionString = ""
    local prevLetter = ""
    -- Iterate through the results of the prediction and append the letter that was predicted in the sample.
    for index, prediction in ipairs(predictions) do
        local maxValue, maxIndex = torch.max(prediction, 1)
        -- Minus 1 to the index because a CTC blank has the index of 0.
        maxIndex = maxIndex[1] - 1
        -- If the index is 0, that means that the character was a CTC blank.
        if (maxIndex ~= 0) then
            local letter = AudioData.findLetter(maxIndex)
            -- We do not add the character if it is the same as the previous character.
            if (letter ~= prevLetter) then
                predictionString = predictionString .. letter
                prevLetter = letter
            end
        end
    end
    local predictedWords = {}
    for word in string.gmatch(predictionString, "%a+") do
        table.insert(predictedWords, SpellingChecker:correct(word))
    end
    local targetWords = {}
    for word in string.gmatch(targetSentence, "%a+") do
        table.insert(targetWords, word)
    end
    return predictedWords, targetWords
end

-- Calculates the word error rate (as a percentage).
function wordErrorRate(target, prediction)
    local d = torch.Tensor(#target + 1, #prediction + 1):zero()
    for i = 1, #target + 1 do
        for j = 1, #prediction + 1 do
            if (i == 1) then
                d[1][j] = j - 1
            elseif (j == 1) then
                d[i][1] = i - 1
            end
        end
    end

    for i = 2, #target + 1 do
        for j = 2, #prediction + 1 do
            if (target[i - 1] == prediction[j - 1]) then
                d[i][j] = d[i - 1][j - 1]
            else
                local substitution = d[i - 1][j - 1] + 1
                local insertion = d[i][j - 1] + 1
                local deletion = d[i - 1][j] + 1
                d[i][j] = torch.min(torch.Tensor({ substitution, insertion, deletion }))
            end
        end
    end
    return d[#target + 1][#prediction + 1] / #target * 100
end

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 128

local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"

--The test set in spectrogram tensor form.
local testDataSet, wordTranscripts = AudioData.retrieveAN4TestDataSet(an4FolderDir, windowSize, stride)

-- File path to the big.txt (see readme for download link). Due to the randomness of the an4 dataset
-- I've combined the transcripts to calculate word probabilities from it. Should be replaced by a proper language model.
SpellingChecker:init("transcriptscombined.txt")

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.model"
}

Network:init(networkParams)
print("Network loaded")

-- We collapse all the words into one large table to pass into the WER calculation.
local totalPredictedWords = {}
local totalTargetWords = {}

-- Loop through the test dataset, getting predictions and target words.
for i = 1, testDataSet:size() do
    local inputs, targets = testDataSet:nextData()
    local predictions = Network:predict(inputs)
    local predictedWords, targetWords = getWords(predictions, wordTranscripts[i])
    for index, word in ipairs(predictedWords) do
        table.insert(totalPredictedWords, word)
    end
    for index, word in ipairs(targetWords) do
        table.insert(totalTargetWords, word)
    end
    xlua.progress(i, testDataSet:size())
end

local rate = wordErrorRate(totalTargetWords, totalPredictedWords)
print(string.format("WER : %.2f percent", rate))
