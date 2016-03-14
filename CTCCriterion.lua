-- Criterion based on the warp-ctc library
-- by Baidu: https://github.com/baidu-research/warp-ctc
-- Formatting of the input can be seen here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
module(..., package.seeall)
require 'nn'
require 'cutorch'
require 'warp_ctc'
require 'CTCBatcher'
local CTCCriterion, parent = torch.class('CTCCriterion', 'nn.Criterion')

function CTCCriterion:__init()
    parent.__init(self)
end

--Calculates and returns the cost of the input respect to the target.
--Example:
--Two input activation sequences: {1,2,3,4,5},{6,7,8,9,10} and {11,12,13,14,15},{16,17,18,19,20}
--Input to the function would be:
--networkOutput: {torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{16,17,18,19,20}})}
--target is the expected labels i.e {{1,2},{3,3}} (for 2 sequences as above).
function CTCCriterion:updateOutput(networkOutput, target)
    local input = convertToCTCBatchSequence(networkOutput)
    local act = input:cuda()
    local grads = torch.CudaTensor()
    local labels = target
    local size = tensorSizes(networkOutput)
    self.output = averageCosts(gpu_ctc(act, grads, labels, size))
    return self.output
end

--Calculates and returns the gradients in respect to the inputs and targets.
--Example:
--Two input activation sequences: {1,2,3,4,5},{6,7,8,9,10} and {11,12,13,14,15},{16,17,18,19,20}
--Input to the function would be:
--networkOutput: {torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{16,17,18,19,20}})}
--target is the expected labels i.e {{1,2},{3,3}} (for 2 sequences as above).
function CTCCriterion:updateGradInput(networkOutput, target)
    local input = convertToCTCBatchSequence(networkOutput)
    local act = input:cuda()
    local temp = nn.SoftMax():updateOutput(input:double()):cuda()
    local grads = temp:clone():zero()
    local labels = target
    local size = tensorSizes(networkOutput)
    gpu_ctc(act, grads, labels, size)
    self.gradInput = convertToNetSequence(grads:float(), #networkOutput)
    return self.gradInput
end

--Returns the largest tensor size and all sizes in a table of tensors
function tensorSizes(tensors)
    return { #tensors }
end


--If batching occurs multiple costs are returned. We sum the costs and return.
function averageCosts(list)
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = acc + v
        end
    end
    return acc
end

return CTCCriterion