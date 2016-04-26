-- Handles the interaction of a fixed size deep neural network of 129 input spectrogram coeffecients and 28 output
-- for speech recognition.
require 'cunn'
require 'cudnn'
require 'optim'
require 'rnn'
require 'nnx'
require 'Linear3D'
require 'TemporalBatchNormalization'
require 'CombineDimensions'
require 'BGRU'
require 'gnuplot'
require 'xlua'

local Network = {}
local logger = optim.Logger('train.log')
logger:setNames { 'loss' }
logger:style { '-' }

function Network:init(networkParams)
    self.loadModel = networkParams.loadModel or false -- If set to true we will load the model into Network.
    self.saveModel = networkParams.saveModel or false -- Set to true if you want to save the model after training.
    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.GRU = networkParams.GRU or false -- Whether to use GRU or LSTM (LSTM by default).
    self.model = nil
    if (self.loadModel) then
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName)
    else
        self:createSpeechNetwork()
    end
    assert((networkParams.saveModel or networkParams.loadModel) and networkParams.fileName, "To save/load you must specify the fileName you want to save to")
end

--Creates a new speech network loaded into Network.
function Network:createSpeechNetwork()
    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(1, 32, 41, 11, 2, 2))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(32, 32, 21, 11, 2, 1))
    model:add(cudnn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.CombineDimensions(2, 3)) -- Combine the middle two dimensions from 4D to 3D (features x batch x seqLength)
    model:add(nn.Transpose({1,2},{2,3})) -- Transpose till batch x seqLength x features

    model:add(nn.TemporalBatchNormalization(32 * 25))
    if (self.GRU) then
        model:add(cudnn.BGRU(32 * 25, 400, 4, true))
    else
        model:add(cudnn.BLSTM(32 * 25, 400, 4, true))
    end
    model:add(nn.TemporalBatchNormalization(800))
    model:add(nn.Linear3D(800, 28))
    model:cuda()
    model:training()
    self.model = model
end

-- Returns a prediction of the input net and input tensors.
function Network:predict(inputTensors)
    local prediction = self.model:forward(inputTensors)
    return prediction
end

--Trains the network using SGD and the defined feval.
--Uses warp-ctc cost evaluation.
function Network:trainNetwork(dataset, epochs, sgd_params)
    local history = {}
    local ctcCriterion = nn.CTCCriterion():cuda()
    local x, gradParameters = self.model:getParameters()

    -- GPU inputs (preallocate)
    local inputs = torch.CudaTensor()

    local function feval(x_new)
        local inputsCPU, targets = dataset:nextData()
        -- transfer over to GPU
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        gradParameters:zero()
        local predictions = self.model:forward(inputs)
        local loss = ctcCriterion:forward(predictions, targets)
        self.model:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local currentLoss
    local startTime = os.time()
    local dataSetSize = dataset:size()
    for i = 1, epochs do
        local averageLoss = 0
        print(string.format("Training Epoch: %d", i))
        for j = 1, dataSetSize do
            currentLoss = 0
            local _, fs = optim.sgd(feval, x, sgd_params)
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, dataSetSize)
            averageLoss = averageLoss + currentLoss
        end
        averageLoss = averageLoss / dataSetSize -- Calculate the average loss at this epoch.
        logger:add { averageLoss } -- Add the average loss value to the logger.
        table.insert(history, averageLoss) -- Add the average loss value to the logger.
        print(string.format("Training Epoch: %d Average Loss: %f", i, averageLoss))
    end
    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    if (self.saveModel) then
        print("Saving model")
        self:saveNetwork(self.fileName)
    end
    return history, minutesTaken
end


function Network:createLossGraph()
    logger:plot()
end


function Network:saveNetwork(saveName)
    torch.save(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName)
    local model = torch.load(saveName)
    self.model = model
    model:evaluate()
end

return Network