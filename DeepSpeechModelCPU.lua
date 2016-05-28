require 'ctchelpers'
require 'rnn'

local function deepSpeech(GRU)
    local model = nn.Sequential()

    model:add(nn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.4))
    model:add(nn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.View(32 * 25, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({2, 3}, {1, 2})) -- seqLength x batch x features

    model:add(nn.SeqBRNN(32 * 25, 400))
    model:add(nn.SeqBRNN(400, 400))
    model:add(nn.SeqBRNN(400, 400))
    model:add(nn.SeqBRNN(400, 400))

    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    model:add(nn.Linear3D(400, 28))
    return model
end

return deepSpeech