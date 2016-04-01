--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain and a simple spell checker.]]

local SpellingChecker = require 'SpellingChecker'
local Network = require 'Network'
local AudioData = require 'AudioData'
require 'nn'
require 'xlua'

--[[Takes the resulting predictions and the transcript sentence. Returns tables of words said in both.]]
local function getWords(predictions, targetSentence, shouldSpellCheck)
    local predictionString = ""
    local prevLetter = ""
    -- Iterate through the results of the prediction and append the letter that was predicted in the sample.
    predictions = predictions:squeeze() -- Remove any single dimensions.
    for x=1,predictions:size(1) do
        local maxValue, maxIndex = torch.max(predictions[x], 1)
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
        if (shouldSpellCheck == true) then
            word = SpellingChecker:correct(word)
        end
        table.insert(predictedWords, word)
    end
    local targetWords = {}
    for word in string.gmatch(targetSentence, "%a+") do
        table.insert(targetWords, word)
    end
    print("expected: " .. targetSentence)
    print("prediction: " ..  predictionString)

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

function calculateWordErrorRate(shouldSpellCheck, testDataSet, wordTranscripts)
    -- We collapse all the words into one large table to pass into the WER calculation.
    local totalPredictedWords = {}
    local totalTargetWords = {}

    for i = 1, #testDataSet do
        local inputAndTarget = testDataSet[i]
        local inputs, targets = inputAndTarget.tensor, inputAndTarget.target
        -- We create an input of size batch x channels x freq x time (batch size in this case is 1).
        inputs = inputs:view(1, 1, inputs:size(1), inputs:size(2)):transpose(3, 4):cuda()
        local predictions = Network:predict(inputs)
        local predictedWords, targetWords = getWords(predictions, wordTranscripts[i], shouldSpellCheck)

        for index, word in ipairs(predictedWords) do
            table.insert(totalPredictedWords, word)
        end
        for index, word in ipairs(targetWords) do
            table.insert(totalTargetWords, word)
        end
        xlua.progress(i, #testDataSet)
    end

    local rate = wordErrorRate(totalTargetWords, totalPredictedWords)
    return rate
end

function fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 75

local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4"

--The test set in spectrogram tensor form.
local testDataSet, wordTranscripts = AudioData.retrieveAN4TestDataSet(an4FolderDir, windowSize, stride)

-- File path to the big.txt (see readme for download link). Due to the randomness of the an4 dataset
-- I've combined the transcripts to calculate word probabilities from it. Should be replaced by a proper language model.
SpellingChecker:init("trainingAN4transcripts.txt")

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.model"
}

Network:init(networkParams)
print("Network loaded")

local spellCheckedWER = calculateWordErrorRate(false, testDataSet, wordTranscripts)
print(string.format("Without Spellcheck WER : %.2f percent", spellCheckedWER))

local spellCheckedWER = calculateWordErrorRate(true, testDataSet, wordTranscripts)
print(string.format("With context based Spellcheck WER : %.2f percent", spellCheckedWER))

-- Make sure that the user has got big.txt else do not carry out the general spellcheck WER.
if (fileExists("big.txt")) then
    -- The final evaluation uses a spell checker conditioned on a large text file of multiple ebooks.
    SpellingChecker:init("big.txt")
    local spellCheckedWER = calculateWordErrorRate(true, testDataSet, wordTranscripts)
    print(string.format("With general Spellcheck WER : %.2f percent", spellCheckedWER))
end

