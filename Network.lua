--Handles the interaction of a fixed size deep neural network of 256 input, 27 output
--for speech recognition.
module(..., package.seeall)
require 'cunn'
require 'CTCCriterion'
require 'optim'
require 'rnn'
require 'gnuplot'

local Network = {}

local logger = optim.Logger('train.log')
logger:setNames { 'loss' }
logger:style { '-' }

--Returns a new network based on the speech recognition stack.
function Network.createSpeechNetwork()
    torch.manualSeed(12345)
    --Used to create the bi-directional RNNs. The fwd is clones to create the bwd
    local fwd = nn.Sequential()
    fwd:add(nn.FastLSTM(300, 300))
    fwd:add(nn.FastLSTM(300, 300))
    fwd:add(nn.FastLSTM(300, 300))
    local net = nn.Sequential()
    net:add(nn.Sequencer(nn.BatchNormalization(129)))
    net:add(nn.Sequencer(nn.TemporalConvolution(129, 584, 5, 1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.TemporalMaxPooling(2, 2)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.BatchNormalization(584)))
    net:add(nn.Sequencer(nn.TemporalConvolution(584, 400, 5, 1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.BatchNormalization(400)))
    net:add(nn.Sequencer(nn.Linear(400, 300)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.BatchNormalization(300)))
    net:add(nn.BiSequencer(fwd))
    net:add(nn.Sequencer(nn.BatchNormalization(600)))
    net:add(nn.Sequencer(nn.Linear(600, 28)))
    return net
end

--Returns the largest tensor size and all sizes in a table of tensors
function findMaxSize(tensors)
    local maxSize = 0
    local allSizes = {}
    for i = 1, #tensors do
        local tensorSize = tensors[i]:size(1)
        if (tensorSize > maxSize) then maxSize = tensorSize end
        table.insert(allSizes, tensorSize)
    end
    return allSizes, maxSize
end

--Pads a dataset with 0's so all tensors are off the same size.
function padDataset(totalInput)
    local allSizes, maxSize = findMaxSize(totalInput)
    local emptyMax = {}
    for i = 1, totalInput[1]:size(2) do
        table.insert(emptyMax, 0)
    end
    for i = 1, #totalInput do
        local input = torch.totable(totalInput[i])
        while (#input < maxSize) do
            table.insert(input, emptyMax)
        end
        totalInput[i] = torch.Tensor(input)
    end
    return totalInput
end

--Creates the dataset depending on the batchSize given. We also pad all the inputs so they are the same size.
function Network.createDataSet(inputJson, labelJson, batchSize)
    local dataset = {}
    for t = 1, #inputJson, batchSize do
        local inputs = {}
        local targets = {}
        for i = t, math.min(t + batchSize - 1, #inputJson) do
            table.insert(inputs, inputJson[i])
            table.insert(targets, labelJson[i])
        end
        table.insert(dataset, { padDataset(inputs), targets })
    end
    local pointer = 1
    --TODO the size of dataset should be #dataset, however to limit to 10 samples I have hard coded this.
    function dataset:size() return 10 end

    function dataset:nextData()
        local sample = dataset[pointer]
        pointer = pointer + 1
        if (pointer > dataset:size()) then pointer = 1 end
        return sample[1], sample[2]
    end

    return dataset
end

--Returns a prediction of the input net and input tensors.
function Network.predict(net, inputTensors)
    local prediction = net:forward(inputTensors)
    return prediction
end

--Trains the network using SGD and the defined feval.
--Uses warp-ctc cost evaluation.
function Network.trainNetwork(net, inputTensors, labels, batchSize, epochs, sgd_params)
    local ctcCriterion = CTCCriterion()
    local x, gradParameters = net:getParameters()
    local dataset = Network.createDataSet(inputTensors, labels, batchSize)
    local function feval(x_new)
        local inputs, targets = dataset:nextData()
        gradParameters:zero()
        local predictions = net:forward(inputs)
        local loss = ctcCriterion:forward(predictions, targets)
        net:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions, targets)
        net:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local currentLoss
    local i = 0
    while i < epochs do
        currentLoss = 0
        i = i + 1
        local _, fs = optim.sgd(feval, x, sgd_params)
        currentLoss = currentLoss + fs[1]
        logger:add { currentLoss }
        print("Loss: ", currentLoss, " iteration: ", i)
    end
end

function Network.createLossGraph()
    logger:plot()
end

return Network