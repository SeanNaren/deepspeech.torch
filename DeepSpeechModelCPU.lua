require 'rnn'
require 'BRNN'
require 'SequenceWise'

local function deepSpeech(GRU)
    local model = nn.Sequential()

    model:add(nn.SpatialConvolution(1, 32, 11, 41, 2, 2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(32, 32, 11, 21, 2, 1))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))

    local rnnInputsize = 32 * 41 -- based on the above convolutions.
    local rnnHiddenSize = 400 -- size of rnn hidden layers
    model:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features


    model:add(nn.BRNN(nn.SeqLSTM(rnnInputsize, rnnHiddenSize)))
    model:add(nn.BRNN(nn.SeqLSTM(rnnHiddenSize, rnnHiddenSize)))
    model:add(nn.BRNN(nn.SeqLSTM(rnnHiddenSize, rnnHiddenSize)))


    local fullConnected = nn.Sequential()
    fullConnected:add(nn.BatchNormalization(rnnHiddenSize))
    fullConnected:add(nn.Linear(rnnHiddenSize, 40))

    model:add(nn.SequenceWise(fullConnected)) -- allows us to maintain 3D structure
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    return model
end

return deepSpeech