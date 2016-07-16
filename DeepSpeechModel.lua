require 'cunn'
require 'cudnn'
require 'SequenceWise'
require 'BatchBRNNReLU'

local function deepSpeech(GRU)

    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(1, 32, 11, 41, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ClippedReLU(true, 20))
    model:add(cudnn.SpatialConvolution(32, 32, 11, 21, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ClippedReLU(true, 20))

    local rnnInputsize = 32 * 41 -- based on the above convolutions.
    local rnnHiddenSize = 1000 -- size of rnn hidden layers
    local nbOfHiddenLayers = 7
    model:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    if (GRU) then
        model:add(cudnn.BGRU(rnnInputsize, rnnHiddenSize, nbOfHiddenLayers))
    else
        for i = 1, nbOfHiddenLayers do
            if i == 1 then
                model:add(cudnn.BatchBRNNReLU(rnnInputsize, rnnHiddenSize))
            else
                model:add(cudnn.BatchBRNNReLU(rnnHiddenSize, rnnHiddenSize))
            end
        end
    end

    local fullConnected = nn.Sequential()
    fullConnected:add(nn.BatchNormalization(rnnHiddenSize))
    fullConnected:add(nn.Linear(rnnHiddenSize, 40))

    model:add(nn.SequenceWise(fullConnected)) -- allows us to maintain 3D structure
    model:add(nn.Transpose({ 1, 2 })) -- batch x seqLength x features
    return model
end

return deepSpeech

