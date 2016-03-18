--Handles the interaction of a fixed size deep neural network of 256 input, 27 output
--for speech recognition.
module(..., package.seeall)
require 'cunn'
require 'cudnn'
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
    local cnn = nn.Sequential()
    cnn:add(nn.BatchNormalization(129))
    cnn:add(nn.TemporalConvolution(129, 256, 5, 1))
    cnn:add(cudnn.ReLU())
    cnn:add(nn.TemporalConvolution(256, 256, 5, 1))
    cnn:add(cudnn.ReLU())
    cnn:add(nn.TemporalMaxPooling(2, 2))
    cnn:add(nn.Dropout(0.25))

    cnn:add(nn.BatchNormalization(256))
    cnn:add(nn.TemporalConvolution(256, 500, 5, 1))
    cnn:add(cudnn.ReLU())
    cnn:add(nn.TemporalConvolution(500, 500, 5, 1))
    cnn:add(cudnn.ReLU())
    cnn:add(nn.TemporalMaxPooling(2, 2))
    cnn:add(nn.Dropout(0.25))

    cnn:add(nn.BatchNormalization(500))
    cnn:add(nn.Linear(500,500))
    cnn:add(cudnn.ReLU())
    cnn:add(nn.Dropout(0.5))


    local model = nn.Sequential()
    model:add(cnn) --  seqlen x inputsize
    model:add(nn.SplitTable(1))
    model:add(nn.BiSequencer(createBiDirectionalNetwork()))
    model:add(nn.Sequencer(nn.Linear(1000, 28)))
    model:cuda()
    return model
end

function createBiDirectionalNetwork()
    local fwd = nn.Sequential()
    fwd:add(nn.FastLSTM(500, 500))
    fwd:add(nn.FastLSTM(500, 500))
    fwd:add(nn.FastLSTM(500, 500))
    fwd:add(nn.FastLSTM(500, 500))
    fwd:add(nn.FastLSTM(500, 500))
    return fwd;
end

--Returns a prediction of the input net and input tensors.
function Network.predict(net, inputTensors)
    local prediction = net:forward(inputTensors)
    return prediction
end

--Trains the network using SGD and the defined feval.
--Uses warp-ctc cost evaluation.
function Network.trainNetwork(net, dataset, epochs, sgd_params)
    local ctcCriterion = CTCCriterion()
    local x, gradParameters = net:getParameters()
    local function feval(x_new)
        local inputs, targets = dataset:nextData()
        inputs = inputs
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
    local startTime = os.time()
    while i < epochs do
        currentLoss = 0
        i = i + 1
        local _, fs = optim.sgd(feval, x, sgd_params)
        currentLoss = currentLoss + fs[1]
        logger:add { currentLoss }
        print("Loss: ", currentLoss, " iteration: ", i)
    end
    local endTime = os.time()
    local secondsTaken = endTime - startTime
    print("Minutes taken to train: ", secondsTaken / 60)
end

function Network.createLossGraph()
    logger:plot()
end

return Network