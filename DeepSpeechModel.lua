require 'cudnn'
require 'ctchelpers'
require 'rnn'
require 'nngraph'

local function deepSpeech(GRU)
    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.View(32 * 25, -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({2, 3}, {1, 2})) -- seqLength x batch x features
    if (GRU) then
        model:add(cudnn.BGRU(32 * 25, 400, 4))
    else
        model:add(cudnn.BLSTM(32 * 25, 400, 4))
    end

    model:add(nn.Transpose({ 1, 2 })) -- batch x seqLength x features
    model:add(nn.MergeConcat(400, 3)) -- Sums the outputDims of the two outputs layers from BRNN into one.
    model:add(nn.Linear3D(400, 28))

    return model
end

return deepSpeech

