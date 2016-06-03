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

local function replace(self, callback)
  local out = callback(self)
  if self.modules then
    for i, module in ipairs(self.modules) do
      self.modules[i] = replace(module, callback)
    end
  end
  return out
end

local function convertBN(net, dst)
    return replace(net, function(x)
        local y = 0
        local src = dst == nn and cudnn or nn
        local src_prefix = src == nn and 'nn.' or 'cudnn.'
        local dst_prefix = dst == nn and 'nn.' or 'cudnn.'

        local function convert(v)
            local y = {}
            torch.setmetatable(y, dst_prefix..v)
            for k,u in pairs(x) do y[k] = u end
            if src == cudnn and x.clearDesc then x.clearDesc(y) end
            return y
        end
        if torch.typename(x) == src_prefix..'BatchNormalization' then
            y = convert('BatchNormalization')
        end
        return y == 0 and x or y
    end)
end

function Network:init(networkParams)

    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.nGPU = networkParams.nGPU
    self.isCUDNN = networkParams.backend == 'cudnn'
    if self.nGPU <= 0 then
        assert(not self.isCUDNN)
    end
    assert(networkParams.batchSize % networkParams.nGPU == 0, 'batch size must be the multiple of nGPU')
    assert(networkParams.validationBatchSize % networkParams.nGPU == 0, 'batch size must be the multiple of nGPU')
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
    self.loadModel = networkParams.loadModel
    self.saveModelIterations = networkParams.saveModelIterations or 10 -- Saves model every number of iterations.

    -- setting model saving/loading
    if (self.loadModel) then
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName,
            networkParams.modelName,
            self.isCUDNN)
    else
        assert(networkParams.modelName, "Must have given a model to train.")
        self:prepSpeechModel(networkParams.modelName)
    end
    assert((networkParams.saveModel or networkParams.loadModel) and networkParams.fileName, "To save/load you must specify the fileName you want to save to")
    -- setting online loading
    self.indexer = indexer(networkParams.trainingSetLMDBPath, networkParams.batchSize)
    self.indexer:prep_sorted_inds()
    self.pool = threads.Threads(1, function() require 'Loader' end)
    self.nbBatches = math.floor(self.indexer.lmdb_size / networkParams.batchSize)

    self.logger = optim.Logger(self.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { 'loss', 'WER' }
    self.logger:style { '-', '-' }
end

function Network:prepSpeechModel(modelName)
    local model = require(modelName)
    self.model = model[1](self.nGPU, self.isCUDNN)
    self.calSizeOfSequences = model[2]
end

function Network:testNetwork(epoch)
    self.model:evaluate()
    if self.
    self.model = convertBN(self.model, nn)
    local wer = self.werTester:getWER(self.nGPU > 0, self.model, self.calSizeOfSequences, true, epoch or 1) -- details in log
    self.model = convertBN(self.model, cudnn)
    self.model:training()
    return wer
end

function Network:trainNetwork(epochs, sgd_params)
    --[[
        train network with self-defined feval (sgd inside); use ctc for evaluation
    --]]
    self.model:training()

    local lossHistory = {}
    local validationHistory = {}
    local x, gradParameters = self.model:getParameters()

    local criterion
    if self.nGPU <= 1 then
        criterion = nn.CTCCriterion()
    end
    if self.nGPU > 0 then
        if self.nGPU == 1 then
            criterion = criterion:cuda()
        end
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
        local inputs, sizes, targets = specBuf, sizesBuf, labelBuf -- move buf to training data
        inds = self.indexer:nxt_sorted_inds() -- load nxt batch
        self.pool:addjob(function()
            return loader:nxt_batch(inds, false)
        end,
            function(spect, label, size)
                specBuf = spect
                labelBuf = label
                sizesBuf = size
            end)
        --------------------- fwd and bwd ---------------------
        -- inputs:resize(inputsCPU:size()):copy(inputsCPU) -- transfer over to GPU
        sizes = self.calSizeOfSequences(sizes)
        if self.nGPU > 0 then
            inputs = inputs:cuda()
            sizes = sizes:cuda()
        end
        local loss
        if criterion then
            local output = self.model:forward({ inputs, sizes })
            loss = criterion:forward(output, targets, sizes)
            local gradOutput = criterion:backward(output, targets)
            self.model:zeroGradParameters()
            self.model:backward(inputs, gradOutput)
        else
            self.model:forward({ inputs, sizes })
            self.model:zeroGradParameters()
            loss = self.model:backward(inputs, targets)
        end
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
            local _, fs = optim.sgd(feval, x, sgd_params)
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
        if self.saveModel and i % self.saveModelIterations == 0 then
            print("Saving model..")
            self:saveNetwork(self.modelTrainingPath .. 'model_epoch_' .. i .. suffix .. '.t7')
        end
    end

    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    if self.saveModel then
        print("Saving model..")
        self:saveNetwork(self.modelTrainingPath .. 'final_model' .. suffix .. '.t7')
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
function Network:loadNetwork(saveName, modelName)
    print ('loading model ' .. saveName)
    local model
    self.prepSpeechModel(model, modelName)
    local weights, gradParameters = self.model:getParameters()
    model = loadDataParallel(saveName, self.nGPU, self.isCUDNN)
    local weights_to_copy, _ = model:getParameters()
    weights:copy(weights_to_copy)
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

return Network
