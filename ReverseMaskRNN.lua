------------------------------------------------------------------------
--[[ ReverseMaskRNN ]] --
-- Applies the same masking procedure as MaskRNN, however reverses the input.
-- Used in bi-directional RNNs for reverse RNN.
-- Input is of dimensions (T*N)*H ( the same as RNN) and output of (T*N)*H.
-- seqLengths: N, indicate the real length of each sample in a minibatch.
------------------------------------------------------------------------
require 'dpnn'
local ReverseMaskRNN, parent = torch.class("nn.ReverseMaskRNN", "nn.Decorator")

function ReverseMaskRNN:__init(module)
    parent.__init(self, module)
end

function ReverseMaskRNN:reverse(input, seqLengths)
    local batchsize = input:size(2)
    assert(batchsize == seqLengths:size(1), ' number of sequence lengths does not match the number of samples in the batch for masking')
    local output = input.new():resizeAs(input):zero()
    for i = 1, batchsize do
        local T = seqLengths[i]
        for t = 1, T do
            output[T - t + 1][i]:copy(input[t][i])
        end
    end
    return output
end

function ReverseMaskRNN:updateOutput(input)
    self.reverse_input = input[1]:view(-1, input[2]:size(1), input[1]:size(2))
    self.reverse_input = self:reverse(self.reverse_input, input[2])
    local reverse_output = self.module:updateOutput(self.reverse_input)
    self.output = self:reverse(reverse_output, input[2])
    self.output = self.output:view(self.reverse_input:size(1) * self.reverse_input:size(2), -1)
    return self.output
end

function ReverseMaskRNN:updateGradInput(input, gradOutput)
    self.reverse_gradOutput = gradOutput:view(self.reverse_input:size(1), input[2]:size(1), -1)
    self.reverse_gradOutput = self:reverse(self.reverse_gradOutput, input[2])
    local reverse_gradInput = self.module:updateGradInput(self.reverse_input,
        self.reverse_gradOutput)
    self.gradInput = self:reverse(reverse_gradInput, input[2]):viewAs(input[1])
    return { self.gradInput, nil }
end

function ReverseMaskRNN:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(self.reverse_input, self.reverse_gradOutput, scale)
end

function ReverseMaskRNN:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(self.reverse_input, self.reverse_gradOutput, lr)
end

function ReverseMaskRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(self.reverse_input, self.reverse_gradOutput, lr)
end
