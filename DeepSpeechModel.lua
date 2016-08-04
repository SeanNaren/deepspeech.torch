require 'UtilsMultiGPU'
require 'SequenceWise'

-- Chooses RNN based on if GRU or backend GPU support.
local function getRNNModule(nIn, nHidden, GRU, is_cudnn)
    if (GRU) then
        if is_cudnn then
            require 'cudnn'
            return cudnn.GRU(nIn, nHidden, 1)
        else
            require 'rnn'
        end
        return nn.GRU(nIn, nHidden)
    end
    if is_cudnn then
        require 'BatchBRNNReLU'
        return cudnn.BatchBRNNReLU(nIn, nHidden)
    else
        require 'rnn'
    end
    return nn.SeqLSTM(nIn, nHidden)
end

local function ReLU(isCUDNN)
    if (isCUDNN) then return cudnn.ClippedReLU(true, 20) else return nn.ReLU(true) end
end

-- Creates the covnet+rnn structure.
local function deepSpeech(nGPU, isCUDNN)
    local model = nn.Sequential()
    if (isCUDNN) then require 'cudnn' end
    local GRU = false
    local conv = nn.Sequential()
    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    conv:add(nn.SpatialConvolution(1, 32, 11, 41, 2, 2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(ReLU(isCUDNN))
    conv:add(nn.SpatialConvolution(32, 32, 11, 21, 2, 1))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(ReLU(isCUDNN))

    local rnnInputsize = 32 * 41 -- based on the above convolutions.
    local rnnHiddenSize = 1760 -- size of rnn hidden layers
    local nbOfHiddenLayers = 7

    conv:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    conv:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    local rnn = nn.Sequential()
    local rnn_module = getRNNModule(rnnInputsize, rnnHiddenSize,
        GRU, isCUDNN)
    rnn:add(rnn_module:clone())
    rnn_module = getRNNModule(rnnHiddenSize,
        rnnHiddenSize, GRU, isCUDNN)

    for i = 1, nbOfHiddenLayers - 1 do
        rnn:add(rnn_module:clone())
    end

    local post_sequential = nn.Sequential()
    post_sequential:add(nn.BatchNormalization(rnnHiddenSize))
    post_sequential:add(nn.Linear(rnnHiddenSize, 28))

    model:add(conv)
    model:add(rnn)
    model:add(nn.SequenceWise(post_sequential))
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    model = makeDataParallel(model, nGPU, isCUDNN)
    return model
end

-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv2
    return sizes
end

return { deepSpeech, calculateInputSizes }