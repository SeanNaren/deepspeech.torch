--[[
--CTCTestCriterion is a non-batch activation function using ctc.
 ]]
module(...,package.seeall)
require 'nn'
require 'warp_ctc'

local CTCTestCriterion, parent = torch.class('CTCTestCriterion','nn.Criterion')

function CTCTestCriterion:__init()
    parent.__init(self)
end

function CTCTestCriterion:updateOutput(input,target)
    local act = torch.FloatTensor(input:size()):copy(input)
    local grads = torch.FloatTensor()
    local labels = {target}
    local size = {act:size(1)}
    self.output = reduce(cpu_ctc(act, grads, labels, size))
    return self.output
end

function CTCTestCriterion:updateGradInput(input,target)
    local act = torch.FloatTensor(input:size()):copy(input)
    local temp = nn.SoftMax():updateOutput(input:double()):float()
    local grads = temp:clone():zero()
    local labels = {target }
    local size = {act:size(1)}
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
    return acc
end