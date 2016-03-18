-- Handles the interaction of a fixed size deep neural network of 129 input spectrogram coeffecients and 28 output
-- for speech recognition.
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

function Network:init(networkParams)
    self.loadModel = networkParams.loadModel or false -- If set to true we will load the model into Network.
    self.saveModel = networkParams.saveModel or false -- Set to true if you want to save the model after training.
    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.model = nil
    if (Network.loadModel) then
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        Network:loadNetwork(networkParams.fileName)
    else
        Network:createSpeechNetwork()
    end
    assert(networkParams.saveModel and networkParams.fileName, "To save you must specify the fileName you want to save to")
end

--Creates a new speech network loaded into Network.
function Network:createSpeechNetwork()
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
    cnn:add(nn.Linear(500, 500))
    cnn:add(cudnn.ReLU())
    cnn:add(nn.Dropout(0.5))

    local model = nn.Sequential()
    model:add(cnn) --  seqlen x inputsize
    model:add(nn.SplitTable(1))
    model:add(createBiDirectionalNetwork())
    model:add(nn.Sequencer(nn.Linear(1000, 28)))
    model:cuda()
    Network.model = model
end

-- Creates the stack of bi-directional RNNs.
function createBiDirectionalNetwork()
    local biseqModel = nn.Sequential()
    biseqModel:add(nn.BiSequencer(nn.FastLSTM(500, 500)))
    biseqModel:add(nn.BiSequencer(nn.FastLSTM(1000, 500)))
    biseqModel:add(nn.BiSequencer(nn.FastLSTM(1000, 500)))
    biseqModel:add(nn.BiSequencer(nn.FastLSTM(1000, 500)))
    biseqModel:add(nn.BiSequencer(nn.FastLSTM(1000, 500)))
    return biseqModel;
end

-- Returns a prediction of the input net and input tensors.
function Network:predict(inputTensors)
    local prediction = Network.model:forward(inputTensors)
    return prediction
end

--Trains the network using SGD and the defined feval.
--Uses warp-ctc cost evaluation.
function Network:trainNetwork(dataset, epochs, sgd_params)
    local ctcCriterion = CTCCriterion()
    local x, gradParameters = Network.model:getParameters()
    local function feval(x_new)
        local inputs, targets = dataset:nextData()
        inputs = inputs
        gradParameters:zero()
        local predictions = Network.model:forward(inputs)
        local loss = ctcCriterion:forward(predictions, targets)
        Network.model:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions, targets)
        Network.model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    -- TODO - one epoch is one iteration of the entire dataset. Update below accordingly.
    local currentLoss
    local i = 0
    local startTime = os.time()
    while i < epochs do
        currentLoss = 0
        i = i + 1
        local _, fs = optim.sgd(feval, x, sgd_params)
        currentLoss = currentLoss + fs[1]
        logger:add { currentLoss }
        -- TODO - consider changing this to use an xlua progress bar for each epoch (loop of training set).
        print("Loss: ", currentLoss, " iteration: ", i)
    end
    local endTime = os.time()
    local secondsTaken = endTime - startTime
    print("Minutes taken to train: ", secondsTaken / 60)

    if (Network.saveModel) then
        print("Saving model")
        Network:saveNetwork(Network.fileName)
    end
end

function Network:createLossGraph()
    logger:plot()
end

function Network:saveNetwork(saveName)
    torch.save(saveName, Network.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName)
    local model = torch.load(saveName)
    Network.model = model
end

return Network