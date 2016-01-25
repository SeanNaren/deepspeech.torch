
require 'nn'
require 'CTCCriterion'
require 'optim'
require 'rnn'
dataset = {};
pointer = 1;
local label1 = {2.0,1.0,4.0} -- {b,a,d}
local label2 = {4.0,1.0,2.0} -- {d,a,b}

for i=1,15 do
    local input1 = torch.rand(86,256)
    local input2 = torch.rand(156,256)
    local tensorInput = {input1,input2}
    dataset[i] = {tensorInput,{label1,label2}}
end
function dataset:size() return 15 end

function dataset:nextData()
    local sample = dataset[pointer]
    pointer = pointer + 1
    if(pointer > dataset:size()) then pointer = 1 end
    return sample[1],sample[2]
end

net = nn.Sequential()
net:add(nn.TemporalConvolution(256,200,1,1))
net:add(nn.ReLU())
net:add(nn.TemporalConvolution(200,150,1,1))
net:add(nn.ReLU())
net:add(nn.TemporalMaxPooling(1,1))
net:add(nn.BatchNormalization(150))
net:add(nn.Linear(150,100))
net:add(nn.ReLU())
net:add(nn.BatchNormalization(100))
net:add(nn.FastLSTM(100,75))
net:add(nn.BatchNormalization(75))
net:add(nn.FastLSTM(75,50))
net:add(nn.BatchNormalization(50))
net:add(nn.FastLSTM(50,40))
net:add(nn.BatchNormalization(40))
net:add(nn.Linear(40,27))
net:add(nn.SoftMax())
net = nn.Sequencer(net)

ctcCriterion = CTCCriterion()
x, gradParameters = net:getParameters()

function feval(params)
    gradParameters:zero()
    local tensorInput,targets = dataset:nextData()
    local f = 0
    for i=1,#tensorInput do
        local prediction = net:forward({tensorInput[i]})
        local loss = ctcCriterion:forward(prediction[1],targets[i])
        f = f + loss
        net:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(prediction[1],targets[i])
        net:backward({tensorInput[i]},{gradOutput})
    end
    gradParameters:div(#tensorInput)
    f = f/#tensorInput
    return f, gradParameters
end

sgd_params = {
    learningRate = 0.1,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9
}

local currentLoss = 90
local iteration = 1
while currentLoss > 385 or iteration < 1000 do
    currentLoss = 0
    iteration = iteration + 1
    local _,fs = optim.sgd(feval,x,sgd_params)
    currentLoss = currentLoss + fs[1]
    print("Loss: ",currentLoss)
end

--print(net:forward(dataset[1][1])[1])