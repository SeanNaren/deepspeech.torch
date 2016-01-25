module(...,package.seeall)
require 'nn'
require 'warp_ctc'

local BatchCTCCriterion, parent = torch.class('BatchCTCCriterion','nn.Criterion')

function BatchCTCCriterion:__init()
    parent.__init(self)
end

function BatchCTCCriterion:updateOutput(inputAndSizes,target)
    local input = inputAndSizes[1]
    local act = torch.FloatTensor(input:size()):copy(input)
    local grads = torch.FloatTensor()
    local labels = target
    local size = inputAndSizes[2]
    self.output = reduce(cpu_ctc(act, grads, labels, size))
    return self.output
end

function BatchCTCCriterion:updateGradInput(inputAndSizes,target)
    local input = inputAndSizes[1]
    local act = torch.FloatTensor(input:size()):copy(input)
    local temp = nn.SoftMax():updateOutput(input:double()):float()
    local grads = temp:clone():zero()
    local labels = target
    local size = inputAndSizes[2]
    cpu_ctc(act,grads,labels,size)
    self.gradInput = torch.DoubleTensor(grads:size()):copy(grads)
    return self.gradInput
end

function reduce(list)
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