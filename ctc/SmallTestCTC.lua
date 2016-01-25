
require 'nn'
require 'CTCCriterion'
require 'optim'
require 'rnn'
dataset = {};
pointer = 1;
local label1 = {3.0,1.0,2.0} -- {c,a,b}
local label2 = {2.0,1.0,3.0} -- {b,a,c}

for i=1,15 do
    local input1 = torch.Tensor({
        {1,2,3,4,5,6,7,8},
        {9,10,11,12,13,14,15,16},
        {17,18,19,20,21,22,23,24}
    })
    local input2 = torch.Tensor({
        {17,18,19,20,21,22,23,24},
        {9,10,11,12,13,14,15,16},
        {1,2,3,4,5,6,7,8}
    })
    dataset[i] = {{input1,input2},{label1,label2}}
end
function dataset:size() return 15 end

function dataset:nextData()
    local sample = dataset[pointer]
    pointer = pointer + 1
    if(pointer > dataset:size()) then pointer = 1 end
    return sample[1],sample[2]
end

local net = nn.Sequential()
net:add(nn.TemporalConvolution(8,7,1,1))
net:add(nn.ReLU())
net:add(nn.TemporalConvolution(7,6,1,1))
net:add(nn.ReLU())
net:add(nn.TemporalMaxPooling(1,1))
net:add(nn.BatchNormalization(6))
net:add(nn.Linear(6,5))
net:add(nn.ReLU())
net:add(nn.BatchNormalization(5))
net:add(nn.FastLSTM(5,5))
net:add(nn.BatchNormalization(5))
net:add(nn.FastLSTM(5,5))
net:add(nn.BatchNormalization(5))
net:add(nn.FastLSTM(5,5))
net:add(nn.BatchNormalization(5))
net:add(nn.Linear(5,4))
net:add(nn.SoftMax())
net = nn.Sequencer(net)

local ctcCriterion = CTCCriterion()
local x, gradParameters = net:getParameters()

function feval(params)
    gradParameters:zero()
    local tensorInput,targets = dataset:nextData()
    local prediction = net:forward(tensorInput)
    local loss = ctcCriterion:forward(prediction,targets)
    net:zeroGradParameters()
    local gradOutput = ctcCriterion:backward(prediction,targets)
    net:backward(tensorInput,gradOutput)
    return loss, gradParameters
end

local sgd_params = {
    learningRate = 0.1,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9
}

local currentLoss = 90
local iteration = 1
while currentLoss > 386 or iteration < 100 do
    currentLoss = 0
    iteration = iteration + 1
    local _,fs = optim.sgd(feval,x,sgd_params)
    currentLoss = currentLoss + fs[1]
    print("Loss: ",currentLoss)
end

print("expected",dataset[2][2])
print(net:forward(dataset[2][1])[1])