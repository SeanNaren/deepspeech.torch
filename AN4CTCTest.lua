--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain and a simple spell checker.]]

local Network = require 'Network'
local Evaluator = require 'Evaluator'
require 'nn'
require 'rnn'

progress = true -- Set to true if you want to see progress of calculation of WER.

function fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = "CTCNetwork.t7",
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 1, -- Number of GPUs, set -1 to use CPU
    lmdb_path = 'prepare_an4/train/',
    val_path = 'prepare_an4/test/',
    batch_size = 1
}

Network:init(networkParams)
print("Network loaded")

test(Network)
