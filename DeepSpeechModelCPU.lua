require 'ctchelpers'
require 'rnn'
require 'BRNN'

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

    model:add(nn.CombineDimensions(2, 3)) -- Combine the middle two dimensions from 4D to 3D (features x batch x seqLength)
    model:add(nn.Transpose({1,2},{2,3})) -- Transpose till batch x seqLength x features

    model:add(nn.BRNN(nn.SeqLSTM(32 * 25, 400)))
    model:add(nn.TemporalBatchNormalization(400))
    model:add(nn.BRNN(nn.SeqLSTM(400, 400)))
    model:add(nn.TemporalBatchNormalization(400))
    model:add(nn.Linear3D(400, 28))
    return model
end

return deepSpeech