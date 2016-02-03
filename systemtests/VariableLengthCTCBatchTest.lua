--[[
--VariableLengthCTCBatchTest takes two variable inputs of size 256
--and correctly labels them to 2 outputs in one batch.
 ]]
require 'nn'
require 'BatchCTCCriterion'
require 'optim'
require 'rnn'
local totalNet = nn.Sequential()
torch.manualSeed(12345)
totalNet:add(nn.Sequencer(nn.TemporalConvolution(256,200,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.TemporalMaxPooling(1,1)))
totalNet:add(nn.Sequencer(nn.TemporalConvolution(200,170,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.TemporalConvolution(170,150,1,1)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.Sequencer(nn.BatchNormalization(150)))
totalNet:add(nn.Sequencer(nn.Linear(150,120)))
totalNet:add(nn.Sequencer(nn.ReLU()))
totalNet:add(nn.BiSequencer(nn.FastLSTM(120,40),nn.FastLSTM(120,40)))
totalNet:add(nn.Sequencer(nn.BatchNormalization(40*2)))
totalNet:add(nn.Sequencer(nn.Linear(40*2,8)))
totalNet:add(nn.Sequencer(nn.SoftMax()))

local input1 = torch.rand(5,256)
local input2 = torch.rand(7,256)
local totalInput = {input1,input2}
local totalTargets = {{2,1,4},{4,1}}

function findMaxSize(tensors)
    local maxSize = 0
    local allSizes = {}
    for i=1,#tensors do
        local tensorSize = tensors[i]:size(1)
        if(tensorSize > maxSize) then maxSize = tensorSize end
        table.insert(allSizes,tensorSize)
    end
    return allSizes,maxSize
end


local allSizes,maxSize = findMaxSize(totalInput)
local emptyMax = {}
for i=1,totalInput[1]:size(2) do
    table.insert(emptyMax,0)
end
for i=1,#totalInput do
    local input = torch.totable(totalInput[i])
    while(#input < maxSize) do
        table.insert(input,emptyMax)
    end
    totalInput[i] = torch.Tensor(input)
end

local ctcCriterion = BatchCTCCriterion()
local x, gradParameters = totalNet:getParameters()
function feval(params)
    gradParameters:zero()
    local tensorInput = totalInput
    local targets = totalTargets
    local allSizes,maxSize = findMaxSize(tensorInput)
    local prediction = totalNet:forward(tensorInput)
    local interleavedPrediction = createInterleavedPrediction(maxSize,prediction)
    local predictionWithSizes = {interleavedPrediction,allSizes}
    local loss = ctcCriterion:forward(predictionWithSizes,targets)
    totalNet:zeroGradParameters()
    local gradOutput = ctcCriterion:backward(predictionWithSizes,targets)
    gradOutput = convertToSequenceData(gradOutput,#totalInput)
    totalNet:backward(tensorInput,gradOutput)
    return loss, gradParameters
end

function convertToSequenceData(gradientOutput, numberOfSequences)
    local gradients = {}
    for i = 1, numberOfSequences do
        table.insert(gradients,{})
    end
    for i = 1, gradientOutput:size(1) do
        local index = math.fmod(i, numberOfSequences)
        if(index == 0) then index = numberOfSequences end
        table.insert(gradients[index],torch.totable(gradientOutput[i]))
    end
    local returnTensors = {}
    for i = 1, numberOfSequences do
        table.insert(returnTensors,torch.Tensor(gradients[i]))
    end
    return returnTensors
end

function createInterleavedPrediction(maxSize,tensors)
    local columnMajor = {}
    for i = 1,maxSize do
        for index,tensor in ipairs(tensors) do
            table.insert(columnMajor,getTensorValue(tensor,i))
        end
    end
    local resultTensor = torch.Tensor(columnMajor)
    return resultTensor
end

function getTensorValue(tensor,index)
    local tensorValue
    if(tensor:size(1) >= index) then tensorValue = tensor[index] end
    if(tensorValue == nil) then
        local emptyTensor = torch.totable(tensor[1]:zero())
        return emptyTensor
    else
        local tableTensorValue = torch.totable(tensorValue)
        return tableTensorValue end
end

local sgd_params = {
    learningRate = 0.01,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9
}

local currentLoss = 300
local iteration = 1
while iteration < 400 do
    currentLoss = 0
    iteration = iteration + 1
    local _,fs = optim.sgd(feval,x,sgd_params)
    currentLoss = currentLoss + fs[1]
    print("Loss: ",currentLoss, " Iteration: ", iteration)
end
print(totalNet:forward(totalInput)[1])
print(totalNet:forward(totalInput)[2])
