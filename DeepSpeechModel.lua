require 'ctchelpers'
require 'rnn'
require 'nngraph'
require 'MaskRNN'
require 'ReverseRNN'
require 'utils_multi_gpu'

function get_rnn_module(nIn, nHidden, GRU, is_cudnn)
    if (GRU) then
        if is_cudnn then
            require 'cudnn'
            return cudnn.GRU(nIn, nHidden, 1)
        end
        return nn.GRU(nIn, nHidden)
    end
    if is_cudnn then
        require 'cudnn'
        return cudnn.LSTM(nIn, nHidden, 1)
    end
    return nn.SeqLSTM(nIn, nHidden)
end

function BRNN(feat, seqLengths, rnn_module)
    local fwdLstm = nn.MaskRNN(rnn_module)({feat, seqLengths})
    local bwdLstm = nn.ReverseRNN(rnn_module)({feat, seqLengths})
    return nn.CAddTable()({fwdLstm, bwdLstm})
end

local function deepSpeech(nGPU, is_cudnn)
    GRU = false
    local seqLengths = nn.Identity()()
    local input = nn.Identity()()
    local feature = nn.Sequential()
    feature:add(nn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    feature:add(nn.SpatialBatchNormalization(32))
    feature:add(nn.ReLU(true))
    feature:add(nn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    feature:add(nn.SpatialBatchNormalization(32))
    feature:add(nn.ReLU(true))
    feature:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    feature:add(nn.View(32 * 25, -1):setNumInputDims(3)) -- batch x features x seqLength
    feature:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features
    local rnn = nn.Identity()({feature(input)})
    rnn_module = get_rnn_module(32 * 25, 400, GRU, is_cudnn)
    rnn = BRNN(rnn, seqLengths, rnn_module)
    rnn_module = get_rnn_module(400, 400, GRU, is_cudnn)
    for i=1,1 do
        rnn = BRNN(rnn, seqLengths, rnn_module)
    end
    local post_sequential = nn.Sequential()
    post_sequential:add(nn.Transpose({ 1, 2 })) -- This should be removed, 
                                                -- as ctc transpose this again.
                                                -- We can use a simpler ctc
    post_sequential:add(nn.MergeConcat(400, 3))
    post_sequential:add(nn.Linear3D(400, 28))
    local model = nn.gModule({input, seqLengths}, {post_sequential(rnn)})
    -- graph.dot(model.fg, 'deepSpeech', 'deepSpeech') -- view graph
    return model
end
return deepSpeech