require 'optim'
require 'nnx'
require 'gnuplot'
require 'lfs'
require 'xlua'
require 'UtilsMultiGPU'
require 'Loader'
require 'nngraph'
require 'Mapper'
require 'ModelEvaluator'

local suffix = '_' .. os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}

--Training parameters
seed = 10
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(seed)

function Network:init(opt)
    self.fileName = opt.modelPath -- The file name to save/load the network from.
    self.nGPU = opt.nGPU
    self.gpu = self.nGPU > 0

    if not self.gpu then
        require 'rnn'
    else
        require 'cutorch'
        require 'cunn'
        require 'cudnn'
        require 'BatchBRNNReLU'
        cutorch.manualSeedAll(seed)
    end
    self.trainingSetLMDBPath = opt.trainingSetLMDBPath
    self.validationSetLMDBPath = opt.validationSetLMDBPath
    self.logsTrainPath = opt.logsTrainPath or nil
    self.logsValidationPath = opt.logsValidationPath or nil
    self.modelTrainingPath = opt.modelTrainingPath or nil

    self:makeDirectories({ self.logsTrainPath, self.logsValidationPath, self.modelTrainingPath })

    self.mapper = Mapper(opt.dictionaryPath)
    self.tester = ModelEvaluator(self.gpu, self.validationSetLMDBPath, self.mapper,
        opt.validationBatchSize, self.logsValidationPath)
    self.saveModel = opt.saveModel
    self.saveModelInTraining = opt.saveModelInTraining or false
    self.loadModel = opt.loadModel
    self.saveModelIterations = opt.saveModelIterations or 10 -- Saves model every number of iterations.
    self.maxNorm = opt.maxNorm or 400 -- value chosen by Baidu for english speech.
    -- setting model saving/loading
    if self.loadModel then
        assert(opt.modelPath, "modelPath hasn't been given to load model.")
        self:loadNetwork(opt.modelPath, opt.modelName)
    else
        assert(opt.modelName, "Must have given a model to train.")
        self:prepSpeechModel(opt.modelName, opt)
    end
    assert((opt.saveModel or opt.loadModel) and opt.modelPath, "To save/load you must specify the modelPath you want to save to")
    -- setting online loading
    self.indexer = indexer(opt.trainingSetLMDBPath, opt.batchSize)
    self.pool = threads.Threads(1, function() require 'Loader' end)

    self.logger = optim.Logger(self.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { 'loss', 'WER', 'CER' }
    self.logger:style { '-', '-', '-' }
end

function Network:prepSpeechModel(modelName, opt)
    local model = require(modelName)
    self.model = model[1](opt)
    self.calSize = model[2]
end

function Network:testNetwork(epoch)
    self.model:evaluate()
    local wer, cer = self.tester:runEvaluation(self.model, true, epoch or 1) -- details in log
    self.model:zeroGradParameters()
    self.model:training()
    return wer, cer
end

function Network:trainNetwork(epochs, optimizerParams)
    self.model:training()

    local lossHistory = {}
    local validationHistory = {}
    local criterion = nn.CTCCriterion(true)
    local x, gradParameters = self.model:getParameters()

    print("Number of parameters: ", gradParameters:size(1))

    -- inputs (preallocate)
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if self.gpu then
        criterion = criterion:cuda()
        inputs = inputs:cuda()
        sizes = sizes:cuda()
    end

    -- def loading buf and loader
    local loader = Loader(self.trainingSetLMDBPath, self.mapper)
    local specBuf, labelBuf, sizesBuf

    -- load first batch
    local inds = self.indexer:nextIndices()
    self.pool:addjob(function()
        return loader:nextBatch(inds)
    end,
        function(spect, label, sizes)
            specBuf = spect
            labelBuf = label
            sizesBuf = sizes
        end)

    -- define the feval
    local function feval(x_new)
        self.pool:synchronize() -- wait previous loading
        local inputsCPU, sizes, targets = specBuf, sizesBuf, labelBuf -- move buf to training data
        inds = self.indexer:nextIndices() -- load next batch whilst training
        self.pool:addjob(function()
            return loader:nextBatch(inds)
        end,
            function(spect, label, sizes)
                specBuf = spect
                labelBuf = label
                sizesBuf = sizes
            end)

        inputs:resize(inputsCPU:size()):copy(inputsCPU) -- transfer over to GPU
        sizes = self.calSize(sizes)
        local predictions = self.model:forward(inputs)
        local loss = criterion:forward(predictions, targets, sizes)
        self.model:zeroGradParameters()
        local gradOutput = criterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        local norm = gradParameters:norm()
        if norm > self.maxNorm then
            gradParameters:mul(self.maxNorm / norm)
        end
        return loss, gradParameters
    end

    -- training
    local currentLoss
    local startTime = os.time()

    for i = 1, epochs do
        local averageLoss = 0
        for j = 1, self.indexer.nbOfBatches do
            currentLoss = 0
            local _, fs = optim.sgd(feval, x, optimizerParams)
            if self.gpu then cutorch.synchronize() end
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.indexer.nbOfBatches)
            averageLoss = averageLoss + currentLoss
        end

        self.indexer:permuteBatchOrder()

        averageLoss = averageLoss / self.indexer.nbOfBatches -- Calculate the average loss at this epoch.

        -- anneal learningRate
        optimizerParams.learningRate = optimizerParams.learningRate / (optimizerParams.learningRateAnnealing or 1)

        -- Update validation error rates
        local wer, cer = self:testNetwork(i)

        print(string.format("Training Epoch: %d Average Loss: %f Average Validation WER: %.2f Average Validation CER: %.2f",
            i, averageLoss, 100 * wer, 100 * cer))

        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer, 100 * cer }

        -- periodically save the model
        if self.saveModelInTraining and i % self.saveModelIterations == 0 then
            print("Saving model..")
            self:saveNetwork(self.modelTrainingPath .. 'model_epoch_' .. i .. suffix .. '_' .. self.fileName)
        end
    end

    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    if self.saveModel then
        print("Saving model..")
        self:saveNetwork(self.modelTrainingPath .. 'final_model_' .. suffix .. '_' .. self.fileName)
    end

    return lossHistory, validationHistory, minutesTaken
end

function Network:createLossGraph()
    self.logger:plot()
end

function Network:saveNetwork(saveName)
    self.model:clearState()
    saveDataParallel(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName, modelName)
    self.model = loadDataParallel(saveName, self.nGPU)
    local model = require(modelName)
    self.calSize = model[2]
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

return Network
