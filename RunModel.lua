local SpellingChecker = require 'SpellingChecker'
local Network = require 'Network'
local AudioData = require 'AudioData'
require 'audio'
require 'nn'
require 'rnn'
require 'xlua'

assert(arg[1], "Must specify filename to AN4 WAV");
local filePath = arg[1]

local an4FolderDir = "/root/CTCSpeechRecognition/Audio/an4/wav/an4test_clstk/" .. filePath
--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 75

os.execute("play " .. an4FolderDir)

local spectrogram = audio.spectrogram(audio.load(an4FolderDir), windowSize, 'hamming', stride):transpose(1, 2)

--[[Takes the resulting predictions and the transcript sentence. Returns tables of words said in both.]]
local function getWords(predictions)
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

            word = SpellingChecker:correct(word)

        table.insert(predictedWords, word)
    end
    return predictedWords, predictionString
end

-- File path to the big.txt (see readme for download link). Due to the randomness of the an4 dataset
-- I've combined the transcripts to calculate word probabilities from it. Should be replaced by a proper language model.
SpellingChecker:init("trainingAN4transcripts.txt")

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.t7",
    gpu = true
}

Network:init(networkParams)

local biggestTensor = spectrogram:size()
local batchTensor = torch.Tensor(1, 1, biggestTensor[1], biggestTensor[2]):transpose(3,4) -- We add 1 dimension (1 feature map).

batchTensor[1] = spectrogram:view(1, biggestTensor[1], biggestTensor[2]):transpose(2,3) -- We add 1 dimension (1 feature map).
batchTensor = batchTensor:cuda()
local predictions = Network:predict(batchTensor)

local words, pureCTCTranscription = getWords(predictions)
local predictionString = ""
for x = 1, #words do
    predictionString = predictionString .. " " .. words[x]
end
print("Predicted Transcription:", pureCTCTranscription)
print("Predicted Transcription with spell checking:", predictionString)
