require 'optim'
require 'nnx'
require 'gnuplot'
require 'lfs'
require 'xlua'
require 'UtilsMultiGPU'
require 'Loader'
require 'nngraph'
require 'Mapper'
require 'WEREvaluator'

local suffix = '_' .. os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}

--Training parameters
seed = 10
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(seed)

function Network:init(networkParams)
    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.nGPU = networkParams.nGPU
    self.gpu = self.nGPU > 0

    if not self.gpu then
        assert(networkParams.backend ~= 'cudnn', "Can't have cuDNN backend when set to CPU mode")
    else
        require 'cutorch'
        require 'cunn'
        cutorch.manualSeedAll(seed)
    end
    self.trainingSetLMDBPath = networkParams.trainingSetLMDBPath
    self.validationSetLMDBPath = networkParams.validationSetLMDBPath
    self.logsTrainPath = networkParams.logsTrainPath or nil
    self.logsValidationPath = networkParams.logsValidationPath or nil
    self.modelTrainingPath = networkParams.modelTrainingPath or nil

    self:makeDirectories({ self.logsTrainPath, self.logsValidationPath, self.modelTrainingPath })

    self.mapper = Mapper(networkParams.dictionaryPath)
    self.werTester = WEREvaluator(self.validationSetLMDBPath, self.mapper, networkParams.validationBatchSize,
        networkParams.validationIterations, self.logsValidationPath)
    self.saveModel = networkParams.saveModel
    self.saveModelInTraining = networkParams.saveModelInTraining or false
    self.loadModel = networkParams.loadModel
    self.saveModelIterations = networkParams.saveModelIterations or 10 -- Saves model every number of iterations.

    -- setting model saving/loading
    if (self.loadModel) then
        require 'SequenceWise'
        require 'BatchBRNNReLU'
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName,
            networkParams.modelName,
            networkParams.backend == 'cudnn')
    else
        assert(networkParams.modelName, "Must have given a model to train.")
        self:prepSpeechModel(networkParams.modelName, networkParams.backend)
    end
    assert((networkParams.saveModel or networkParams.loadModel) and networkParams.fileName, "To save/load you must specify the fileName you want to save to")
    -- setting online loading
    self.indexer = indexer(networkParams.trainingSetLMDBPath, networkParams.batchSize)
    self.indexer:prep_sorted_inds()
    self.pool = threads.Threads(1, function() require 'Loader' end)
    self.nbBatches = math.ceil(self.indexer.lmdb_size / networkParams.batchSize)

    self.logger = optim.Logger(self.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { 'loss', 'WER' }
    self.logger:style { '-', '-' }
end

function Network:prepSpeechModel(modelName, backend)
    local model = require(modelName)
    self.model = model[1](self.nGPU, backend == 'cudnn')
    self.calSize = model[2]
end

function Network:testNetwork(epoch)
    self.model:evaluate()
    local wer = self.werTester:getWER(self.gpu, self.model, true, epoch or 1) -- details in log
    self.model:zeroGradParameters()
    self.model:training()
    return wer
end

local function synchronize(self)
    if self.gpu then cutorch.synchronize() end
end

function Network:trainNetwork(epochs, sgd_params)
    --[[
        train network with self-defined feval (sgd inside); use ctc for evaluation
    --]]
    self.model:training()

    local lossHistory = {}
    local validationHistory = {}
    local ctcCriterion = nn.CTCCriterion(true)
    local x, gradParameters = self.model:getParameters()

    print("Number of parameters: ", gradParameters:size(1))

    -- inputs (preallocate)
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if self.gpu then
        ctcCriterion = ctcCriterion:cuda()
        inputs = inputs:cuda()
        sizes = sizes:cuda()
    end

    -- def loading buf and loader
    local loader = Loader(self.trainingSetLMDBPath)
    local specBuf, labelBuf, sizesBuf

    -- load first batch
    local inds = self.indexer:nxt_sorted_inds()
    self.pool:addjob(function()
        return loader:nxt_batch(inds, false)
    end,
        function(spect, label, sizes)
            specBuf = spect
            labelBuf = label
            sizesBuf = sizes
        end)

    -- define the feval
    local function feval(x_new)
        --------------------- data load ------------------------
        self.pool:synchronize() -- wait previous loading
        local inputsCPU, sizes, targets = specBuf, sizesBuf, labelBuf -- move buf to training data
        inds = self.indexer:nxt_sorted_inds() -- load nxt batch
        self.pool:addjob(function()
            return loader:nxt_batch(inds, false)
        end,
            function(spect, label, sizes)
                specBuf = spect
                labelBuf = label
                sizesBuf = sizes
            end)
        --------------------- fwd and bwd ---------------------
        inputs:resize(inputsCPU:size()):copy(inputsCPU) -- transfer over to GPU
        sizes = self.calSize(sizes)
        local predictions = self.model:forward(inputs)
        local loss = ctcCriterion:forward(predictions, targets, sizes)
        self.model:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        gradParameters:div(inputs:size(1))
        return loss, gradParameters
    end

    -- training
    local currentLoss
    local startTime = os.time()

    for i = 1, epochs do
        local averageLoss = 0
        for j = 1, self.nbBatches do
            currentLoss = 0
            synchronize(self)
            local _, fs = optim.sgd(feval, x, sgd_params)
            synchronize(self)
            if self.model.needsSync then
                self.model:syncParameters()
            end
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.nbBatches)
            averageLoss = averageLoss + currentLoss
        end

        averageLoss = averageLoss / self.nbBatches -- Calculate the average loss at this epoch.

        -- Update validation error rates
        local wer = self:testNetwork(i)

        print(string.format("Training Epoch: %d Average Loss: %f Average Validation WER: %.2f%%", i, averageLoss, 100 * wer))
        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer }

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
    saveDataParallel(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName, modelName, is_cudnn)
    self.model = loadDataParallel(saveName, self.nGPU, is_cudnn)
    local model = require(modelName)
    self.calSize = model[2]
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

return Network
