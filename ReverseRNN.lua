------------------------------------------------------------------------
--[[ ReverseRNN ]]--
-- Input and Output size: T*N*H ( the same as RNN)
-- seqLengths: N, indicate the real length of each sample in a minibatch
------------------------------------------------------------------------
require 'dpnn'
local ReverseRNN, parent = torch.class("nn.ReverseRNN", "nn.Decorator")

function ReverseRNN:__init(module)
   parent.__init(self, module)
end

function ReverseRNN:reverse(input, seqLengths)
   local batchsize = input:size(2)
   assert(batchsize == seqLengths:size(1))
   local output = input.new():resizeAs(input):zero()
   for i=1,batchsize do
      local T = seqLengths[i]
      for t=1,T do
         output[T-t+1][i]:copy(input[t][i])
      end
   end
   return output
end

function ReverseRNN:updateOutput(input)
   self.reverse_input = self:reverse(input[1], input[2])
   reverse_output = self.module:updateOutput(self.reverse_input)
   self.output = self:reverse(reverse_output, input[2])
   return self.output
end

function ReverseRNN:updateGradInput(input, gradOutput)
   self.reverse_gradOutput = self:reverse(gradOutput, input[2])
   reverse_gradInput = self.module:updateGradInput(self.reverse_input,
   													self.reverse_gradOutput)
   self.gradInput = self:reverse(reverse_gradInput, input[2])
   return {self.gradInput, nil}
end

function ReverseRNN:accGradParameters(input, gradOutput, scale) 
   self.module:accGradParameters(self.reverse_input, self.reverse_gradOutput, scale)
end

function ReverseRNN:accUpdateGradParameters(input, gradOutput, lr)
   self.module:accUpdateGradParameters(self.reverse_input, self.reverse_gradOutput, lr)
end

function ReverseRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
   self.module:sharedAccUpdateGradParameters(self.reverse_input, self.reverse_gradOutput, lr)
end
