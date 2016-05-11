------------------------------------------------------------------------
--[[ ReverseRNN ]]--
-- Input and Output size: N*T*H ( the same as RNN)
-- seqLengths: N, indicate the real length of each sample in a minibatch
------------------------------------------------------------------------
local ReverseRNN, parent = torch.class("nn.ReverseRNN", "nn.Decorator")

function ReverseRNN:__init(module)
   parent.__init(self, module)
end

function ReverseRNN:reverse(input, seqLengths)

   return input
end

function ReverseRNN:updateOutput(input)
   self.reverse_input = self:reverse(input[1], seqLengths)
   reverse_output = self.module:updateOutput(self.reverse_input)
   self.output = self:reverse(reverse_output, seqLengths)
   return self.output
end

function ReverseRNN:updateGradInput(input, gradOutput)
   reverse_gradOutput = self:reverse(gradOutput, seqLengths)
   reverse_gradInput = self.module:updateGradInput(self.reverse_input,
   													reverse_gradOutput)
   self.gradInput = self:reverse(reverse_gradInput, seqLengths)
   return {self.gradInput, nil}
end
