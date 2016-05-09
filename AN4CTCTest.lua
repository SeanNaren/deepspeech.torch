--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain and a simple spell checker.]]

local SpellingChecker = require 'SpellingChecker'
local Network = require 'Network'
local AudioData = require 'AudioData'
local WERCalculator = require 'WERCalculator'
require 'nn'
require 'rnn'
require 'xlua'

gpu = true -- Set to true if you trained a GPU based model.
progress = true -- Set to true if you want to see progress of calculation of WER.

function fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 75

local an4FolderDir = "/data/LLL/data/speech/an4"

--The test set in spectrogram tensor form.
-- local testDataSet = AudioData.retrieveAN4TestDataSet(an4FolderDir, windowSize, stride)

-- File path to the big.txt (see readme for download link). Due to the randomness of the an4 dataset
-- I've combined the transcripts to calculate word probabilities from it. Should be replaced by a proper language model.
SpellingChecker:init("trainingAN4transcripts.txt")

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.t7",
    modelName = 'DeepSpeechModel',
    gpu = true, -- Set this to false to revert back to CPU.
    lmdb_path = '/data1/yuanyang/torch_projects/data/an4_lmdb_test/',
    batch_size = 1
}

Network:init(networkParams)
print("Network loaded")

-- test iteration. Since batch size is 1 so iter should be the same as test set size
test_iter = 130
-- TODO test/dict lmdb path
_dir = '/data1/yuanyang/torch_projects/data/an4_lmdb_test/'
dict_path = 'dictionary'

assert(#_dir>1 and #dict_path>1, 'set dir and dict_path first')

local spellCheckedWER = WERCalculator.calculateWordErrorRate(false, test_iter, SpellingChecker, Network.model, gpu, _dir, dict_path)
print(string.format("Without Spellcheck WER : %.2f percent", spellCheckedWER))

local spellCheckedWER = WERCalculator.calculateWordErrorRate(true, test_iter, SpellingChecker, Network.model, gpu, _dir, dict_path)
print(string.format("With context based Spellcheck WER : %.2f percent", spellCheckedWER))

-- Make sure that the user has got big.txt else do not carry out the general spellcheck WER. -- TODO replace with Seq2Seq
if (fileExists("big.txt")) then
    -- The final evaluation uses a spell checker conditioned on a large text file of multiple ebooks.
    SpellingChecker:init("big.txt")
    local spellCheckedWER = WERCalculator.calculateWordErrorRate(true, test_iter, SpellingChecker, Network.model, gpu, _dir, dict_path)
    print(string.format("With general Spellcheck WER : %.2f percent", spellCheckedWER))
end

