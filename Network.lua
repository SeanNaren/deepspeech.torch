require 'optim'
require 'nnx'
require 'BRNN'
require 'gnuplot'
require 'xlua'
local PERCalculator = require 'PERCalculator'

local Network = {}
local logger = optim.Logger('train.log')
logger:setNames { 'loss', 'PER' }
logger:style { '-', '-' }

function Network:init(networkParams)
    self.loadModel = networkParams.loadModel or false -- Set to true to load the model into Network.
    self.saveModel = networkParams.saveModel or false -- Set to true if you want to save the model after training.
    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.gpu = networkParams.gpu or false -- Set to true to use GPU.
    self.model = nil
    self.validation = networkParams.validation or nil
    if (self.gpu) then -- Load gpu modules.
    require 'cunn'
    require 'cudnn'
    end
    if (self.loadModel) then
        require 'SequenceWise'
        require 'BatchBRNNReLU'
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName)
    else
        assert(networkParams.model, "Must have given a model to train.")
        self:prepSpeechModel(networkParams.model)
    end
    assert((networkParams.saveModel or networkParams.loadModel) and networkParams.fileName, "To save/load you must specify the fileName you want to save to")
end

function Network:prepSpeechModel(model)
    if (self.gpu) then model:cuda() end
    model:training()
    self.model = model
end

-- Returns a prediction of the input net and input tensors.
function Network:predict(inputTensors)
    local prediction = self.model:forward(inputTensors)
    return prediction
end

local function calculateValidationPER(self, validationDataset)
    if (validationDataset) then
        self.model:evaluate()
        local per = PERCalculator.calculateValidationPER(validationDataset, self.gpu, self.model)
        self.model:zeroGradParameters()
        self.model:training()
        return per
    end
    return 0
end

--Trains the network using SGD and the defined feval.
--Uses warp-ctc cost evaluation.
function Network:trainNetwork(dataset, epochs, sgd_params, validationDataset)
    local lossHistory = {}
    local validationHistory = {}
    local ctcCriterion = nn.CTCCriterion(true)

    local x, gradParameters = self.model:getParameters()

    print("Number of parameters: ", x:size(1))

    -- inputs (preallocate)
    local inputs = torch.Tensor()
    if self.gpu then
        ctcCriterion = nn.CTCCriterion(true):cuda()
        inputs = inputs:cuda()
    end

    local function feval(x_new)
        local inputsCPU, targets = dataset:nextData()
        -- transfer over to GPU
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        gradParameters:zero()
        local predictions = self.model:forward(inputs)
        local sizes = torch.Tensor(predictions:size(1)):fill(predictions:size(2))
        local loss = ctcCriterion:forward(predictions, targets, sizes)
        self.model:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local currentLoss
    local startTime = os.time()
    local dataSetSize = dataset:size()
    local per = 1
    for i = 1, epochs do
        local averageLoss = 0

        print(string.format("Training Epoch: %d", i))

        -- Periodically update validation error rates
        if (i % 2 == 0) then
            per = calculateValidationPER(self, validationDataset)
            if per then table.insert(validationHistory, 100 * per) end
        end

        for j = 1, dataSetSize do
            currentLoss = 0
            local _, fs = optim.sgd(feval, x, sgd_params)
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, dataSetSize)
            averageLoss = averageLoss + currentLoss
        end

        averageLoss = averageLoss / dataSetSize -- Calculate the average loss at this epoch.
        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        print(string.format("Training Epoch: %d Average Loss: %f PER: %.0f%%", i, averageLoss, 100 * per))

        logger:add { averageLoss, 1000 * per } -- Phone Error Rate
    end
    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    if (self.saveModel) then
        print("Saving model")
        self:saveNetwork(self.fileName)
    end
    return lossHistory, validationHistory, minutesTaken
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