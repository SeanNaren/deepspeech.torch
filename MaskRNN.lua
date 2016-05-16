------------------------------------------------------------------------
--[[ MaskRNN ]]--
-- filter out outputs and grads of unnecessary timesteps to support
-- variant lengths in minibatch
-- Input and Output size: T*N*H ( the same as RNN)
-- seqLengths: N, indicate the real length of each sample in a minibatch
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
   for i=1,batchsize do
      if seqLengths[i] < T then
         input:sub(seqLengths[i]+1, T, i, i):zero()
      end
   end
end

function MaskRNN:updateOutput(input)
   self.output = self.module:updateOutput(input[1])
   self:filter(self.output, input[2])
   return self.output
end

function MaskRNN:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input[1], gradOutput)
   self:filter(self.gradInput, input[2])
   return {self.gradInput, nil}
end

function MaskRNN:accGradParameters(input, gradOutput, scale) 
   self.module:accGradParameters(input[1], gradOutput, scale)
end

function MaskRNN:accUpdateGradParameters(input, gradOutput, lr)
   self.module:accUpdateGradParameters(input[1], gradOutput, lr)
end

function MaskRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
   self.module:sharedAccUpdateGradParameters(input[1], gradOutput, lr)
end
