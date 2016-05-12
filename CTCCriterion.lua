require 'cutorch'
require 'warp_ctc'
------------------------------------------------------------------------
--[[ CTCCriterion ]]--
-- CTC Alignment for sequence data where input and labels do not align.
-- Useful for speech recognition on a phoneme/character level basis.
-- Inputs assumed are in the form of batch x time x inputdim.
-- Targets assumed in the form of {{1,2},{3,4}} where {1,2} is for the first
-- element.
------------------------------------------------------------------------
local CTCCriterion, parent = torch.class('nn.CTCCriterionTest', 'nn.Criterion')

CTCCriterion.dim = 2

function CTCCriterion:__init()
    parent.__init(self)
end

local function nan(data)
    assert(data:ne(data):sum() == 0)
end
function CTCCriterion:forward(input, target, sizes)
    return self:updateOutput(input, target, sizes)
end

function CTCCriterion:updateOutput(input, labels, sizes)
    local acts = input
    if acts:dim() == 3 then
        acts:view(acts, acts:size(1)*acts:size(2), -1)
    end
    assert(acts:nDimension() == CTCCriterion.dim)
    self.sizes = torch.totable(sizes)
    self.gradInput = input.new():resizeAs(input):zero()
    if (input:type() == 'torch.CudaTensor') then
        self.output = sumCosts(gpu_ctc(acts, self.gradInput, labels, self.sizes))
    else
        self.output = sumCosts(cpu_ctc(acts:float(), self.gradInput:float(), labels, self.sizes))
    end
    return self.output / sizes:size(1)
end

function CTCCriterion:updateGradInput(input, labels)
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