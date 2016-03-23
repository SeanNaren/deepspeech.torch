-- Criterion based on the warp-ctc library
-- by Baidu: https://github.com/baidu-research/warp-ctc
-- Formatting of the input can be seen here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
module(..., package.seeall)
require 'nn'
require 'cutorch'
require 'warp_ctc'
local CTCCriterion, parent = torch.class('CTCCriterion', 'nn.Criterion')

function CTCCriterion:__init()
    parent.__init(self)
end

--Calculates and returns the cost of the input respect to the target.
--Example:
--Two input activation sequences: {1,2,3,4,5} and {11,12,13,14,15}
--Input to the function would be:
--networkOutput: {torch.Tensor({1,2,3,4,5}),torch.Tensor({11,12,13,14,15})}
--target is the expected labels i.e {{1,2},{3,3}} (for 2 sequences as above).
function CTCCriterion:updateOutput(networkOutput, target)
    local act = CTCCriterion.convertToCTCSequence(networkOutput):cuda()
    local grads = torch.CudaTensor()
    local labels = target
    local size = {#networkOutput}
    self.output = averageCosts(gpu_ctc(act, grads, labels, size))
    return self.output
end

--Calculates and returns the gradients in respect to the inputs and targets.
--Example:
--Two input activation sequences: {1,2,3,4,5} and {11,12,13,14,15}
--Input to the function would be:
--networkOutput: {torch.Tensor({1,2,3,4,5}),torch.Tensor({11,12,13,14,15})}
--target is the expected labels i.e {{1,2},{3,3}} (for 2 sequences as above).
function CTCCriterion:updateGradInput(networkOutput, target)
    local act = CTCCriterion.convertToCTCSequence(networkOutput):cuda()
    local temp = nn.SoftMax():updateOutput(act:double()):cuda()
    local grads = temp:clone():zero()
    local labels = target
    local size = {#networkOutput}
    gpu_ctc(act, grads, labels, size)
    self.gradInput = CTCCriterion.convertToNetSequence(grads:float(), #networkOutput)
    return self.gradInput
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

--input to function: {torch.Tensor({1,2,3,4,5}),torch.Tensor({11,12,13,14,15})}
--Returned batched format: torch.Tensor({{1,2,3,4,5},{11,12,13,14,15}})
function CTCCriterion.convertToCTCSequence(tensors)
    local columnMajor = {}
    for index, tensor in ipairs(tensors) do
        table.insert(columnMajor, torch.totable(tensor))
    end
    local resultTensor = torch.Tensor(columnMajor)
    return resultTensor
end

--Reverses the batching process to give the gradientOutput for
--backwards propagation on the net.
--Example:
--input to function (output of the CTCCriterion):
--torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
--Returned format:
--{torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{0,0,0,0,0}})}
function CTCCriterion.convertToNetSequence(gradientOutput, numberOfSequences)
    local gradients = {}
    for i = 1, gradientOutput:size(1) do
        local index = math.fmod(i, numberOfSequences)
        if (index == 0) then index = numberOfSequences end
        gradients[index] =  torch.totable(gradientOutput[i])
    end
    local returnTensors = {}
    for i = 1, numberOfSequences do
        table.insert(returnTensors, torch.CudaTensor(gradients[i]))
    end
    return returnTensors
end

return CTCCriterion