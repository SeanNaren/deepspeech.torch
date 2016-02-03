--[[
--LargeSequenceClassificationCTCTest takes three large sequences of 256 length,
--and correctly classifies them to 3 labels.
 ]]
require 'nn'
require 'cunn'
require 'CTCTestCriterion'
require 'optim'
require 'rnn'
local net = nn.Sequential()
torch.manualSeed(12345)
net:add(nn.Sequencer(nn.TemporalConvolution(256,256,5,1)))
net:add(nn.Sequencer(nn.ReLU()))
net:add(nn.Sequencer(nn.TemporalMaxPooling(2,2)))
net:add(nn.Sequencer(nn.TemporalConvolution(256,256,5,1)))
net:add(nn.Sequencer(nn.ReLU()))
net:add(nn.Sequencer(nn.TemporalConvolution(256,256,5,1)))
net:add(nn.Sequencer(nn.ReLU()))

local input1 = torch.rand(120,256)
local input2 = torch.rand(256,120)
local input3 = torch.rand(256,60)

print(net:forward({input1}))