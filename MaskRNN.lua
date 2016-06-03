------------------------------------------------------------------------
--[[ MaskRNN ]] --
-- Filter out outputs and grads of unnecessary timesteps to support
-- variable lengths in minibatch.
-- Input is of dimensions (T*N)*H ( the same as RNN) and output of (T*N)*H.
-- seqLengths: N, indicate the real length of each sample in a minibatch.
------------------------------------------------------------------------
require 'dpnn'

local MaskRNN, parent = torch.class("nn.MaskRNN", "nn.Decorator")

function MaskRNN:__init(module)
    parent.__init(self, module)
    assert(torch.isTypeOf(module, 'nn.Module'))
end

function MaskRNN:filter(input, seqLengths)
    local batchsize = input:size(2)
    assert(batchsize == seqLengths:size(1))
    local T = input:size(1)
    for i = 1, batchsize do
        if seqLengths[i] < T then
            input:sub(seqLengths[i] + 1, T, i, i):zero()
        end
    end
end

function MaskRNN:updateOutput(input)
    self._input = input[1]:view(-1, input[2]:size(1), input[1]:size(2))
    self.output = self.module:updateOutput(self._input)
    self:filter(self.output, input[2])
    self.output = self.output:view(-1, self.output:size(3))
    return self.output
end

function MaskRNN:updateGradInput(input, gradOutput)
    self._gradOutput = gradOutput:view(self._input:size(1), input[2]:size(1), -1)
    self.gradInput = self.module:updateGradInput(self._input, self._gradOutput)
    self:filter(self.gradInput, input[2])
    self.gradInput = self.gradInput:viewAs(input[1])
    return { self.gradInput, nil }
end

function MaskRNN:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(self._input, self._gradOutput, scale)
end

function MaskRNN:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(self._input, self._gradOutput, lr)
end

function MaskRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(self._input, self._gradOutput, lr)
end
