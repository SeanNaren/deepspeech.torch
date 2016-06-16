require 'BatchRNN'
------------------------------------------------------------------------
--[[ BatchBRNNReLU ]] --
-- Based On BatchBRNN. Adds ClippedReLU non-linearity to Vanilla BRNN.
------------------------------------------------------------------------
local BatchRNNReLU, parent = torch.class('cudnn.BatchRNNReLU', 'cudnn.BatchRNN')

function BatchRNNReLU:__init(inputDim, outputDim)
    parent.__init(self, inputDim, outputDim)
    local rnn = self.rnn
    rnn.mode = 'CUDNN_RNN_RELU'
    rnn:reset()
    self:insert(cudnn.ClippedReLU(20, true), 6)
end