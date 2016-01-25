module(...,package.seeall)
require 'nn'
require 'BatchCTCCriterion'
require 'optim'
require 'rnn'

local Network = {}

function Network.createNewNetwork()
    local totalNet = nn.Sequential()
    torch.manualSeed(12345)
    totalNet:add(nn.Sequencer(nn.TemporalConvolution(256,200,1,1)))
    totalNet:add(nn.Sequencer(nn.ReLU()))
    totalNet:add(nn.Sequencer(nn.TemporalMaxPooling(2,2)))
    totalNet:add(nn.Sequencer(nn.TemporalConvolution(200,170,1,1)))
    totalNet:add(nn.Sequencer(nn.ReLU()))
    totalNet:add(nn.Sequencer(nn.TemporalConvolution(170,150,1,1)))
    totalNet:add(nn.Sequencer(nn.ReLU()))
    totalNet:add(nn.Sequencer(nn.BatchNormalization(150)))
    totalNet:add(nn.Sequencer(nn.Linear(150,120)))
    totalNet:add(nn.Sequencer(nn.ReLU()))
    totalNet:add(nn.BiSequencer(nn.FastLSTM(120,40),nn.FastLSTM(120,40)))
    totalNet:add(nn.Sequencer(nn.BatchNormalization(40*2)))
    totalNet:add(nn.Sequencer(nn.Linear(40*2,27)))
    totalNet:add(nn.Sequencer(nn.SoftMax()))
    return totalNet
end

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

function padDataset(totalInput)
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
    return totalInput
end

function createDataSet(inputJson, labelJson)
    local dataset = {}
    local batchSize = 10
    for t = 1,#inputJson,batchSize do
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+batchSize-1,#inputJson) do
            table.insert(inputs, inputJson[i])
            table.insert(targets, labelJson[i])
        end
        table.insert(dataset,{padDataset(inputs),targets})
    end
    local pointer = 1
    function dataset:size() return #dataset end
    function dataset:nextData()
        local sample = dataset[pointer]
        pointer = pointer + 1
        if(pointer > dataset:size()) then pointer = 1 end
        return sample[1],sample[2]
    end
    return dataset
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


function Network.predict(net,inputTensors)
    local prediction = net:forward({inputTensors})[1]
    local results = torch.totable(prediction)
    return results
end

function Network.trainNetwork(net, jsonInputs, jsonLabels)
    local ctcCriterion = BatchCTCCriterion()
    local x, gradParameters = net:getParameters()
    local dataset = createDataSet(jsonInputs, jsonLabels)
    local function feval(params)
        local inputs,targets = dataset:nextData()
        gradParameters:zero()
        local allSizes,maxSize = findMaxSize(inputs)
        local prediction = net:forward(inputs)
        local interleavedPrediction = createInterleavedPrediction(maxSize,prediction)
        local predictionWithSizes = {interleavedPrediction,allSizes }

        local loss = ctcCriterion:forward(predictionWithSizes,targets)
        net:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictionWithSizes,targets)
        gradOutput = convertToSequenceData(gradOutput,#inputs)
        net:backward(inputs,gradOutput)
        return loss, gradParameters
    end

    local sgd_params = {
        learningRate = 0.0001,
        learningRateDecay = 1e-9,
        weightDecay = 0,
        momentum = 0.9
    }
    local currentLoss = 1000
    local i = 0
    while i < 1000 do
        currentLoss = 0
        i = i + 1
        local _,fs = optim.sgd(feval,x,sgd_params)
        currentLoss = currentLoss + fs[1]
        print("Loss: ",currentLoss, " iteration: ", i)
    end

end
return Network