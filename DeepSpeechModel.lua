-- require 'ctchelpers'
require 'rnn'
require 'nngraph'
require 'MaskRNN'
require 'ReverseRNN'
require 'utils_multi_gpu'

local function get_rnn_module(nIn, nHidden, GRU, is_cudnn)
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

local function BRNN(feat, seqLengths, rnn_module)
    local fwdLstm = nn.MaskRNN(rnn_module:clone())({feat, seqLengths})
    local bwdLstm = nn.ReverseRNN(rnn_module:clone())({feat, seqLengths})
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
    local rnn_nFeature_input = 32 * 25
    local rnn_nFeature_output = 400
    feature:add(nn.View(rnn_nFeature_input, -1):setNumInputDims(3)) -- batch x features x seqLength
    feature:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features
    feature:add(nn.View(-1, rnn_nFeature_input))   -- (seqLength x batch) x features
    local rnn = nn.Identity()({feature(input)})
    rnn_module = get_rnn_module(rnn_nFeature_input, rnn_nFeature_output,
                                GRU, is_cudnn)
    rnn = BRNN(rnn, seqLengths, rnn_module)
    rnn_module = get_rnn_module(rnn_nFeature_output,
                                rnn_nFeature_output, GRU, is_cudnn)
    for i=1,0 do
        rnn = nn.Sequential():add(nn.BatchNormalization(rnn_nFeature_output))(rnn)
        rnn = BRNN(rnn, seqLengths, rnn_module)
    end
    local post_sequential = nn.Sequential()
    post_sequential:add(nn.BatchNormalization(rnn_nFeature_output))
    post_sequential:add(nn.Linear(rnn_nFeature_output, 28))
    local model = nn.gModule({input, seqLengths}, {post_sequential(rnn)})
    model = makeDataParallel(model, nGPU, is_cudnn)
    return model
end

local function calSize(sizes)
    sizes = torch.floor((sizes-41)/2+1) -- conv1
    sizes = torch.floor((sizes-21)/2+1) -- conv2
    sizes = torch.floor((sizes-2)/2+1) -- pool1
    return sizes
end

return {deepSpeech, calSize}