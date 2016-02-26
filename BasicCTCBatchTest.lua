--[[
--BasicCTCBatchTest takes 2 fixed inputs of 5 length, and
 - correctly labels them to 2 labels in one batch.
 ]]
require 'cunn'
require 'CTCCriterion'
require 'CTCBatcher'
require 'optim'
require 'rnn'

--Given the prediction returns the numerical labels that the network has predicted.
function turnToNumericalLabels(outputFromNetwork)
    local entries = torch.totable(outputFromNetwork)
    local predictions = {}
    for unusedIndex, entry in ipairs(entries) do
        local maxIndex = 1
        local maxValue = 0
        for index, value in ipairs(entry) do
            if (value > maxValue) then maxValue = value maxIndex = index end
        end
        --We minus 1 off the label to give us the correct positioning.
        table.insert(predictions,maxIndex - 1)
    end
    return predictions
end

--We create two inputs of length 3 to act as our input audio.
--local input1 = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}})
--local input2 = torch.Tensor({{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
local input1 = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}})
local input2 = torch.Tensor({{16,17,18,19,20},{21,22,23,24,25},{0,0,0,0,0}})
local totalInput = {input1,input2 }
--These are the labels of each inputs (BAD and DA)
local totalTargets = {{2,1,4},{6,5}}

--Create a smaller refined network that takes a 5 input and returns 27 labels
local net = nn.Sequential()
torch.manualSeed(123456)
net:add(nn.Sequencer(nn.BatchNormalization(5)))
net:add(nn.Sequencer(nn.TemporalConvolution(5,4,1,1)))
net:add(nn.Sequencer(nn.ReLU()))
net:add(nn.Sequencer(nn.BatchNormalization(4)))
net:add(nn.Sequencer(nn.TemporalConvolution(4,4,1,1)))
net:add(nn.Sequencer(nn.ReLU()))
net:add(nn.Sequencer(nn.TemporalMaxPooling(1,1)))
net:add(nn.Sequencer(nn.BatchNormalization(4)))
net:add(nn.Sequencer(nn.Linear(4,4)))
net:add(nn.Sequencer(nn.ReLU()))
net:add(nn.BiSequencer(nn.LSTM(4,3),nn.LSTM(4,3)))
net:add(nn.Sequencer(nn.BatchNormalization(3*2)))
net:add(nn.Sequencer(nn.Linear(3*2,27)))
net:add(nn.Sequencer(nn.SoftMax()))

--The feval evaluation used by SGD using the CTC criterion.
local ctcCriterion = CTCCriterion()
local x, gradParameters = net:getParameters()
function feval(x_new)
    local inputs, targets = totalInput,totalTargets
    gradParameters:zero()
    local predictions = net:forward(inputs)
    local loss = ctcCriterion:forward(predictions, targets)
    net:zeroGradParameters()
    local gradOutput = ctcCriterion:backward(predictions, targets)
    net:backward(inputs, gradOutput)
    return loss, gradParameters
end

--Parameters we use for training.
local sgd_params = {
    learningRate = 0.1,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.9
}

local currentLoss
local iteration = 1
while iteration < 400 do
    currentLoss = 0
    iteration = iteration + 1
    local _,fs = optim.sgd(feval,x,sgd_params)
    currentLoss = currentLoss + fs[1]
    print("Loss: ",currentLoss, " Iteration: ", iteration)
end

--Predicts the label of the training data (counter intuitive for practical use but able to see if learning has taken place).
print("duplicates labels may occur due to the nature of the CTC function.")
print("input1",turnToNumericalLabels(net:forward(totalInput)[1]))
print("input2",turnToNumericalLabels(net:forward(totalInput)[2]))
