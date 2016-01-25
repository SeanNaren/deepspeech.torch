-- Criterion based on the warp-ctc library
-- by Baidu: https://github.com/baidu-research/warp-ctc
-- Formatting of the input can be seen here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
module(...,package.seeall)
require 'nn'
require 'warp_ctc'

local CTCCriterion, parent = torch.class('CTCCriterion','nn.Criterion')

function CTCCriterion:__init()
    parent.__init(self)
end

--Calculates and returns the cost of the input respect to the target.
--The input and size of each sequence has to be batched together within a table.
--The input has to be in the format specified at the end of this README:
--https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
--Example:
--Two input activation sequences: {1,2,3,4,5},{6,7,8,9,10} and {11,12,13,14,15},{16,17,18,19,20}
--Input to the function would be:
--inputData: torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{16,17,18,19,20}})
--sizes would be: {2,2}
--And the final input to the function would be would be: {inputData,sizes}
--target is the expected labels i.e {{1,2},{3,3}} (for 2 sequences).
function CTCCriterion:updateOutput(inputAndSizes,target)
    local input = inputAndSizes[1]
    local act = torch.FloatTensor(input:size()):copy(input)
    local grads = torch.FloatTensor()
    local labels = target
    local size = inputAndSizes[2]
    self.output = averageCosts(cpu_ctc(act, grads, labels, size))
    return self.output
end

--Calculates and returns the gradients in respecto the inputs and targets.
--The input and size of each sequence has to be batched together within a table.
--The input has to be in the format specified at the end of this README:
--https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
--Example:
--Two input activation sequences: {1,2,3,4,5},{6,7,8,9,10} and {11,12,13,14,15},{16,17,18,19,20}
--Input to the function would be:
--inputData: torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{16,17,18,19,20}})
--sizes would be: {2,2}
--And the final input to the function would be would be: {inputData,sizes}
--target is the expected labels i.e {{1,2},{3,3}} (for 2 sequences).
function CTCCriterion:updateGradInput(inputAndSizes,target)
    local input = inputAndSizes[1]
    local act = torch.FloatTensor(input:size()):copy(input)
    local temp = nn.SoftMax():updateOutput(input:double()):float()
    local grads = temp:clone():zero()
    local labels = target
    local size = inputAndSizes[2]
    cpu_ctc(act,grads,labels,size)
    self.gradInput = grads:float()
    return self.gradInput
end

--If batching occurs multiple costs are returned. We average them and return 1 value.
function averageCosts(list)
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = acc +  v
        end
    end
    return acc / #list
end