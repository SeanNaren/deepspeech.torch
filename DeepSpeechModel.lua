require 'cudnn'
require 'ctchelpers'
require 'rnn'

local function deepSpeech(GRU)

    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.SplitTable(1)) -- batchsize x featuremap x freq x time
    model:add(nn.Sequencer(nn.View(1, 32 * 25, -1))) -- features x freq x time
    model:add(nn.JoinTable(1)) -- batch x features x time
    model:add(nn.Transpose({ 2, 3 })) -- batch x time x features
    model:add(nn.TemporalBatchNormalization(32 * 25))
    model:add(nn.Transpose({ 1, 2 })) -- time x batch x features

    if (GRU) then
        model:add(cudnn.BGRU(32 * 25, 400, 4))
    else
        model:add(cudnn.BLSTM(32 * 25, 400, 4))
    end

    model:add(nn.Transpose({ 1, 2 })) -- batch x seqLength x features
    model:add(nn.MergeConcat(400, 3)) -- Sums the outputDims of the two outputs layers from BRNN into one.
    model:add(nn.TemporalBatchNormalization(400))
    model:add(nn.Linear3D(400, 28))
    return model
end

return deepSpeech

