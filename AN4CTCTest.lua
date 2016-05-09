--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain and a simple spell checker.]]

local SpellingChecker = require 'SpellingChecker'
local Network = require 'Network'
local WERCalculator = require 'WERCalculator'
require 'nn'
require 'rnn'
require 'xlua'

progress = true -- Set to true if you want to see progress of calculation of WER.

function fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

--The test set in spectrogram tensor form.

-- File path to the big.txt (see readme for download link). Due to the randomness of the an4 dataset
-- I've combined the transcripts to calculate word probabilities from it. Should be replaced by a proper language model.
SpellingChecker:init("trainingAN4transcripts.txt")

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.t7",
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 1, -- Number of GPUs, set -1 to use CPU
    lmdb_path = 'prepare_an4/test/',
    batch_size = 1
}

gpu = networkParams.nGPU > 0

Network:init(networkParams)
print("Network loaded")

-- test iteration. Since batch size is 1 so iter should be the same as test set size
test_iter = 130
-- TODO test/dict lmdb path
_dir = networkParams.lmdb_path
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

