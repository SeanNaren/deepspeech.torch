require 'ctchelpers'
require 'rnn'

local function deepSpeech(nGPU)

    GRU = false

    local model = nn.Sequential()
    model:add(nn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    if nGPU > 0 then
        model:add(nn.SplitTable(1)) -- batchsize x featuremap x freq x time
        model:add(nn.Sequencer(nn.View(1, 32 * 25, -1))) -- features x freq x time
        model:add(nn.JoinTable(1)) -- batch x features x time
        model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- batch x time x features
        require 'cudnn'
        if (GRU) then
            model:add(cudnn.BGRU(32 * 25, 400, 4))
        else
            model:add(cudnn.BLSTM(32 * 25, 400, 4))
        end
        model:add(nn.Transpose({ 1, 2 })) -- batch x seqLength x features
        model:add(nn.MergeConcat(400, 3)) -- Sums the outputDims of the two outputs layers from BRNN into one.
    else
        model:add(nn.CombineDimensions(2, 3)) -- Combine the middle two dimensions from 4D to 3D (features x batch x seqLength)
        model:add(nn.Transpose({1,2},{2,3})) -- Transpose till batch x seqLength x features
        require 'BRNN'
        model:add(nn.BRNN(nn.SeqLSTM(32 * 25, 400)))
        model:add(nn.TemporalBatchNormalization(400))
        model:add(nn.BRNN(nn.SeqLSTM(400, 400)))
        model:add(nn.TemporalBatchNormalization(400))
    end

    model:add(nn.Linear3D(400, 28))
    model = makeDataParallel(model, nGPU)
    return model
end

return deepSpeech

