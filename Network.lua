--Handles the interaction of a fixed size deep neural network of 256 input, 27 output
--for speech recognition.
module(..., package.seeall)
require 'cunn'
require 'warp_ctc'
require 'CTCCriterionGPU'
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
    local fwd = createBiDirectionalNetwork()
    local net = nn.Sequential()

    net:add(nn.Sequencer(nn.BatchNormalization(129)))
    net:add(nn.Sequencer(nn.TemporalConvolution(129, 384, 5, 1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.TemporalMaxPooling(2, 2)))
    net:add(nn.Sequencer(nn.BatchNormalization(384)))
    net:add(nn.Sequencer(nn.TemporalConvolution(384, 600, 5, 1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.BatchNormalization(600)))
    net:add(nn.Sequencer(nn.TemporalConvolution(600, 700, 5, 1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.BatchNormalization(700)))
    net:add(nn.Sequencer(nn.TemporalConvolution(700, 700, 5, 1)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.TemporalMaxPooling(2, 2)))

    net:add(nn.Sequencer(nn.BatchNormalization(700)))
    net:add(nn.Sequencer(nn.Linear(700, 700)))
    net:add(nn.Sequencer(nn.ReLU()))
    net:add(nn.Sequencer(nn.BatchNormalization(700)))
    net:add(nn.Sequencer(nn.Linear(700, 700)))
    net:add(nn.Sequencer(nn.BatchNormalization(700)))
    net:add(nn.BiSequencer(fwd))
    net:add(nn.Sequencer(nn.BatchNormalization(1400)))
    net:add(nn.Sequencer(nn.Linear(1400, 700)))
    net:add(nn.Sequencer(nn.BatchNormalization(700)))
    net:add(nn.Sequencer(nn.Linear(700, 28)))
    net:cuda()
    return net
end

function createBiDirectionalNetwork()
    local fwd = nn.Sequential()
    fwd:add(nn.FastLSTM(700, 700))
    fwd:add(nn.FastLSTM(700, 700))
    fwd:add(nn.FastLSTM(700, 700))
    fwd:add(nn.FastLSTM(700, 700))
    fwd:add(nn.FastLSTM(700, 700))
    fwd:add(nn.FastLSTM(700, 700))
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
    local ctcCriterion = CTCCriterionGPU()
    local x, gradParameters = net:getParameters()
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