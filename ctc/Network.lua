--Handles the interaction of a fixed size deep neural network of 256 input, 27 output
--for speech recognition.
module(...,package.seeall)
require 'cunn'
require 'CTCCriterion'
require 'optim'
require 'rnn'
require 'gnuplot'

local Network = {}

local logger = optim.Logger('train.log')
logger:setNames{'loss'}
logger:style{'-'}

--Returns a new network based on the speech recognition stack.
function Network.createSpeechNetwork()

    local fwd = nn.Sequential()
    fwd:add(nn.Sequencer(nn.FastLSTM(300)))
    fwd:add(nn.Sequencer(nn.FastLSTM(300)))
    fwd:add(nn.Sequencer(nn.FastLSTM(300)))
    fwd:add(nn.Sequencer(nn.FastLSTM(300)))
    fwd:add(nn.Sequencer(nn.FastLSTM(300)))

    local net = nn.Sequential()
    net:add(nn.Sequencer(nn.TemporalConvolution(129,129,5,1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.TemporalMaxPooling(2,2)))
    net:add(nn.Sequencer(nn.TemporalConvolution(129,384,5,1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.TemporalMaxPooling(2,2)))
    net:add(nn.Sequencer(nn.TemporalConvolution(384,384,5,1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.Linear(384,300)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.BiSequencerLM(fwd))
    net:add(nn.Sequencer(nn.Linear(300*2,27)))
    net:add(nn.Sequencer(nn.SoftMax()))
    return net
end

--Returns the largest tensor size and all sizes in a table of tensors
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

--Pads a dataset with 0's so all tensors are off the same size.
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

function Network.createDataSet(inputJson, labelJson, batchSize)
    local dataset = {}
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

--Returns a prediction of the input net and input tensors.
function Network.predict(net,inputTensors)
    local prediction = net:forward(inputTensors)
    return prediction
end

--Trains the network using SGD and the defined feval.
--Uses warp-ctc cost evaluation.
function Network.trainNetwork(net, inputTensors, labels, batchSize, epochs)
    local ctcCriterion = CTCCriterion()
    local x, gradParameters = net:getParameters()
    local dataset = Network.createDataSet(inputTensors, labels, batchSize)

    local function feval(x_new)
        local inputs,targets = dataset:nextData()
        gradParameters:zero()
        local predictions = net:forward(inputs)
        local loss = ctcCriterion:forward(predictions,targets)
        net:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions,targets)
        net:backward(inputs,gradOutput)
        return loss, gradParameters
    end

    local sgd_params = {
        learningRate = 10e-4,
        learningRateDecay = 1e-9,
        weightDecay = 0,
        momentum = 0.9
    }
    local currentLoss
    local i = 0
    while i < epochs  do
        currentLoss = 0
        i = i + 1
        local _,fs = optim.sgd(feval,x,sgd_params)
        currentLoss = currentLoss + fs[1]
        logger:add{currentLoss}
        print("Loss: ",currentLoss, " iteration: ", i)
    end
end

function Network.createLossGraph()
    logger:plot()
end

return Network