--[[
--MediumClassificationCTCTest takes 2 fixed inputs of 256 length, and
 - correctly labels them to 2 labels.
 ]]
require 'nn'
require 'CTCTestCriterion'
require 'optim'
require 'rnn'
local totalNet = nn.Sequential()
torch.manualSeed(12345)
totalNet:add(nn.Sequencer(nn.TemporalConvolution(256,200,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.TemporalConvolution(200,150,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.TemporalMaxPooling(1,1)))
totalNet:add(nn.Sequencer(nn.BatchNormalization(150)))
totalNet:add(nn.Sequencer(nn.Linear(150,120)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.BiSequencer(nn.FastLSTM(120,40),nn.FastLSTM(120,40)))
totalNet:add(nn.Sequencer(nn.BatchNormalization(40*2)))
totalNet:add(nn.Sequencer(nn.Linear(40*2,8)))
totalNet:add(nn.Sequencer(nn.SoftMax()))

local input1 = torch.rand(3,256)
local input2 = torch.rand(3,256)
local totalInput = {input1,input2}
local totalTargets = {{2,1,4},{4,1,2}}

local ctcCriterion = CTCTestCriterion()
local x, gradParameters = totalNet:getParameters()
local counter = 1
function feval(params)
    gradParameters:zero()
    local tensorInput = totalInput[counter]
    local targets = totalTargets[counter]
    if counter == #totalInput then counter = 1 else counter = counter + 1 end
    local prediction = totalNet:forward({tensorInput})
    local loss = ctcCriterion:forward(prediction[1],targets)
    totalNet:zeroGradParameters()
    local gradOutput = ctcCriterion:backward(prediction[1],targets)
    totalNet:backward({tensorInput},{gradOutput})
    return loss, gradParameters
end

local sgd_params = {
    learningRate = 0.1,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.9
}

local currentLoss = 200
local iteration = 1
while iteration < 200 do
    currentLoss = 0
    iteration = iteration + 1
    local _,fs = optim.sgd(feval,x,sgd_params)
    currentLoss = currentLoss + fs[1]
    print("Loss: ",currentLoss, " Iteration: ", iteration)
end

print(totalNet:forward({totalInput[1]})[1])
