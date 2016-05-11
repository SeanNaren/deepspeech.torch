------------------------------------------------------------------------
--[[ MaskRNN ]]--
-- filter out outputs and grads of unnecessary timesteps to support
-- variant lengths in minibatch
-- Input and Output size: N*T*H ( the same as RNN)
-- seqLengths: N, indicate the real length of each sample in a minibatch
------------------------------------------------------------------------
local MaskRNN, parent = torch.class("nn.MaskRNN", "nn.Decorator")

function MaskRNN:__init(module)
   parent.__init(self, module)
   assert(torch.isTypeOf(module, 'nn.Module'))
end

function MaskRNN:filter(input, seqLengths)
   assert(input:size(2) == seqLengths:size(1))
   return input
end

function MaskRNN:updateOutput(input)
   self.output = self.module:updateOutput(input[1])
   self.output = self:filter(self.output, input[2])
   return self.output
end

function MaskRNN:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input[1], gradOutput)
   self.gradInput = self:filter(self.gradInput, input[2])
   return {self.gradInput, nil}
end
