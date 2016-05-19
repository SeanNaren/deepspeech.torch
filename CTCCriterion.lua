------------------------------------------------------------------------
--[[ CTCCriterion ]] --
-- CTC Alignment for sequence data where input and labels do not align.
-- Useful for speech recognition on a phoneme/character level basis.
-- Inputs assumed are in the form of seqLength x batch x inputDim.
-- If batchFirst = true then input in the form of batch x seqLength x inputDim.
-- Targets assumed in the form of {{1,2},{3,4}} where {1,2} is for the first
-- element and so forth.
------------------------------------------------------------------------
local CTCCriterion, parent = torch.class('nn.CTCCriterionTest', 'nn.Criterion')

function CTCCriterion:__init(batchFirst)
    require 'warp_ctc'
    parent.__init(self)
    self.acts = torch.Tensor()
    self.batchFirst = batchFirst or false
end

function CTCCriterion:forward(input, target, sizes)
    return self:updateOutput(input, target, sizes)
end

function CTCCriterion:updateOutput(input, target, sizes)
    assert(sizes,
        "You must pass the size of each sequence in the batch as a tensor")
    local acts = input
    if input:dim() == 3 then
        acts:resizeAs(input):copy(input)
        if self.batchFirst then
            acts = acts:transpose(1, 2)
            acts = self:makeContiguous(acts)
        end
        acts:view(acts, acts:size(1) * acts:size(2), -1)
    end
    assert(acts:nDimension() == 2)
    self.sizes = torch.totable(sizes)
    self.gradInput = acts.new():resizeAs(acts):zero()
    if input:type() == 'torch.CudaTensor' then
        self.output = sumCosts(gpu_ctc(acts, self.gradInput, target, self.sizes))
    else
        self.gradInput = self.gradInput:float()
        self.output = sumCosts(cpu_ctc(acts:float(), self.gradInput, target, self.sizes))
    end
    return self.output / sizes:size(1)
end

function CTCCriterion:updateGradInput(input, target)
    if input:dim() == 2 then -- (seqLen * batchSize) x outputDim
        return self.gradInput
    end
    if self.batchFirst then -- batchSize x seqLen x outputDim
        self.gradInput = inverseInterleave(self.gradInput, input:size(1))
    else -- seqLen x batchSize x outputDim
        self.gradInput:view(self.gradInput, input:size(1), input:size(2), -1)
    end
    return self.gradInput
end

function CTCCriterion:makeContiguous(input)
    if not input:isContiguous() then
        self._input = self._input or input.new()
        self._input:typeAs(input):resizeAs(input):copy(input)
        input = self._input
    end
    return input
end

function inverseInterleave(tensor, batchSize)
    local sizes = torch.LongStorage(3)
    sizes[1] = batchSize
    sizes[2] = tensor:size(1) / batchSize
    sizes[3] = tensor:size(2)
    local result = tensor.new():resize(sizes):zero()
    local counter = 1
    for i = 1, sizes[2] do
        for j = 1, sizes[1] do
            result[j][i]:copy(tensor[counter])
            counter = counter + 1
        end
    end
    return result
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