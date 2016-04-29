require 'cudnn'
require 'ctchelpers'
require 'rnn'

local function deepSpeech(GRU)

    local model = nn.Sequential()
    model:add(cudnn.SpatialBatchNormalization(1))
    model:add(cudnn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.SplitTable(1)) -- batchsize x featuremap x freq x seqLength
    model:add(nn.Sequencer(nn.View(1, 32 * 25, -1))) -- features x freq x seqLength
    model:add(nn.JoinTable(1)) -- batch x features x seqLength

    model:add(nn.TemporalBatchNormalization(32 * 25)) -- Keep a running mean on the features dim.
    model:add(nn.Transpose({ 1, 3 }, { 2, 3 })) -- seqLength x batch x features

    if (GRU) then
        model:add(cudnn.BGRU(32 * 25, 400, 4))
    else
        model:add(cudnn.BLSTM(32 * 25, 400, 4))
    end

    model:add(nn.Transpose({ 1, 2 })) -- batch x seqLength x features * 2
    model:add(nn.MergeConcat(400, 3)) -- batch x seqLength x features
    model:add(nn.Transpose({1, 2})) -- seqLength x batch x features
    model:add(nn.View(-1, 400)) -- seqLength*batch x features
    model:add(nn.BatchNormalization(400))
    model:add(nn.Linear(400, 28))
    return model
end

return deepSpeech

