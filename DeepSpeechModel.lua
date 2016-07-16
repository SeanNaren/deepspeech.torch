require 'cunn'
require 'cudnn'
require 'SequenceWise'

local function deepSpeech(GRU)

    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    local rnnInputsize = 32 * 25 -- based on the above convolutions.
    local rnnHiddenSize = 1300 -- size of rnn hidden layers
    local nbOfHiddenLayers = 8
    model:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    if (GRU) then
        model:add(cudnn.BGRU(rnnInputsize, rnnHiddenSize, nbOfHiddenLayers))
    else
        model:add(cudnn.BLSTM(rnnInputsize, rnnHiddenSize, nbOfHiddenLayers))
    end

    model:add(nn.View(-1, 2, rnnHiddenSize):setNumInputDims(2)) -- sum the outgoing weights
    model:add(nn.Sum(3))
    local fullConnected = nn.Sequential()
    fullConnected:add(nn.BatchNormalization(rnnHiddenSize))
    fullConnected:add(nn.Linear(rnnHiddenSize, 28))

    model:add(nn.SequenceWise(fullConnected)) -- allows us to maintain 3D structure
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    return model
end

return deepSpeech

