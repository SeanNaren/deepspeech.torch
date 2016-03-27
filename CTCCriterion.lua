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

function CTCCriterion:updateOutput(output, labels)
    local tensorSizes = output:size()
    local acts = CTCCriterion.createCTCBatch(output, tensorSizes)
    local sizes = {}
    for x = 1, tensorSizes[1] do
        table.insert(sizes, tensorSizes[2])
    end
    local grads = torch.CudaTensor()
    self.output = sumCosts(gpu_ctc(acts, grads, labels, sizes))
    return self.output
end


function CTCCriterion:updateGradInput(output, labels)
    local tensorSizes = output:size()
    local acts = CTCCriterion.createCTCBatch(output, tensorSizes)
    local sizes = {}
    for x = 1, tensorSizes[1] do
        table.insert(sizes, tensorSizes[2])
    end
    local grads = acts:clone():zero()
    gpu_ctc(acts, grads, labels, sizes)
    self.gradInput = CTCCriterion.revertBatching(grads, tensorSizes)
    return self.gradInput
end

--If batching occurs multiple costs are returned. We sum the costs and return.
function sumCosts(list)
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

-- GPU inputs (preallocate)
local acts = torch.CudaTensor()
local convertedGradients = torch.CudaTensor()

function CTCCriterion.createCTCBatch(output, sizes)
    acts:resize(sizes[1] * sizes[2], sizes[3]):zero()
    local counter = 1
    for i = 1, sizes[2] do
        for j = 1, sizes[1] do
            acts[counter] = output[j][i]
            counter = counter + 1
        end
    end
    return acts
end

function CTCCriterion.revertBatching(gradients, sizes)
    convertedGradients:resize(sizes[1] ,sizes[2], sizes[3]):zero()
    local counter = 1
    for i = 1, sizes[2] do
        for j = 1, sizes[1] do
            convertedGradients[j][i] = gradients[counter]
            counter = counter + 1
        end
    end
    return convertedGradients
end