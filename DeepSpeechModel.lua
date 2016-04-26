require 'cudnn'
require 'ctchelpers'

local function deepSpeech(GRU)
    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.CombineDimensions(2, 3)) -- Combine the middle two dimensions from 4D to 3D (features x batch x seqLength)
    model:add(nn.Transpose({1,3})) -- Transpose till seqLength x batch x features

    model:add(nn.TemporalBatchNormalization(32 * 25))
    if (GRU) then
        model:add(cudnn.BGRU(32 * 25, 200, 2))
    else
        model:add(cudnn.BLSTM(32 * 25, 200, 2))
    end
    model:add(nn.Transpose({1,2})) -- batch x seqLength x features
    model:add(nn.MergeConcat(200,3)) -- Sums the outputDims of the two outputs layers from BRNN into one.
    model:add(nn.TemporalBatchNormalization(200))
    model:add(nn.Linear3D(200, 28))
    return model
end

return deepSpeech