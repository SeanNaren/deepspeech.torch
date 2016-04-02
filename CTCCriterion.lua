-- Criterion based on the warp-ctc library
-- by Baidu: https://github.com/baidu-research/warp-ctc
-- Formatting of the input can be seen here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
require 'warp_ctc'

local CTCCriterion, parent = torch.class('nn.CTCCriterion', 'nn.Criterion')

function CTCCriterion:__init()
    parent.__init(self)
    -- GPU inputs (preallocate)
    self.acts = torch.Tensor()
    self.convertedGradients = torch.Tensor()
end

function CTCCriterion:updateOutput(output, labels)
    local tensorSizes = output:size()
    local acts = self:createCTCBatch(output, tensorSizes)
    local sizes = {}
    for x = 1, tensorSizes[1] do
        table.insert(sizes, tensorSizes[2])
    end
    if (output:type() == 'torch.CudaTensor') then
        local grads = torch.CudaTensor()
        self.output = sumCosts(gpu_ctc(acts, grads, labels, sizes))
    else
        local grads = torch.DoubleTensor()
        self.output = sumCosts(cpu_ctc(acts, grads, labels, sizes))
    end

    return self.output
end


function CTCCriterion:updateGradInput(output, labels)
    local tensorSizes = output:size()
    local acts = self:createCTCBatch(output, tensorSizes)
    local sizes = {}
    for x = 1, tensorSizes[1] do
        table.insert(sizes, tensorSizes[2])
    end
    local grads = acts:clone():zero()
    if (output:type() == 'torch.CudaTensor') then
        gpu_ctc(acts, grads, labels, sizes)
    else
        cpu_ctc(acts, grads, labels, sizes)
    end
    self.gradInput = self:revertBatching(grads, tensorSizes)
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

function CTCCriterion:createCTCBatch(output, sizes)
    self.acts:resize(sizes[1] * sizes[2], sizes[3]):zero()
    local counter = 1
    for i = 1, sizes[2] do
        for j = 1, sizes[1] do
            self.acts[counter] = output[j][i]
            counter = counter + 1
        end
    end
    return self.acts
end

function CTCCriterion:revertBatching(gradients, sizes)
    self.convertedGradients:resize(sizes[1], sizes[2], sizes[3]):zero()
    local counter = 1
    for i = 1, sizes[2] do
        for j = 1, sizes[1] do
            self.convertedGradients[j][i] = gradients[counter]
            counter = counter + 1
        end
    end
    return self.convertedGradients
end