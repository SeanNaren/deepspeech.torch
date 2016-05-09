require 'optim'
require 'nnx'
require 'BRNN'
require 'ctchelpers'
require 'gnuplot'
require 'xlua'
require 'utils_multi_gpu'
require 'loader'
local threads = require 'threads'

local WERCalculator = require 'WERCalculator'
local Network = {}
local logger = optim.Logger('train.log')
logger:setNames { 'loss', 'WER' }
logger:style{'-', '-'}


function Network:init(networkParams)
    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.nGPU = networkParams.nGPU
    self.lmdb_path = networkParams.lmdb_path
    -- setting model saving/loading
    if (self.loadModel) then
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName)
    else
        assert(networkParams.modelName, "Must have given a model to train.")
        self:prepSpeechModel(networkParams.modelName, networkParams.backend)
    end
    if self.nGPU > 0 then
        self.model:cuda()
        if networkParams.backend == 'cudnn' then
            require 'cudnn'
            cudnn.fastest = true
            cudnn.convert(self.model, cudnn)
        end
    end
    print (self.model)
    self.model:training()
    assert((networkParams.saveModel or networkParams.loadModel) and networkParams.fileName, "To save/load you must specify the fileName you want to save to")
    -- setting online loading
    self.indexer = indexer(networkParams.lmdb_path, networkParams.batch_size)
    self.pool = threads.Threads(1,function() require 'loader' end)
end

function Network:prepSpeechModel(modelName, backend)
    local model = require (modelName)
    self.model = model(self.nGPU, backend)
end

-- Returns a prediction of the input net and input tensors.
function Network:predict(inputTensors)
    local prediction = self.model:forward(inputTensors)
    return prediction
end

local function WERValidationSet(self, validationSet)
    if(validationSet) then
        self.model:evaluate()
        local wer =  WERCalculator.calculateWordErrorRate(false, validationSet, nil, self.model, self.nGPU > 0)
        self.model:zeroGradParameters()
        self.model:training()
        return wer
    end
end

function Network:trainNetwork(epochs, sgd_params)
    --[[
        train network with self-defined feval (sgd inside); use ctc for evaluation
    --]]

    local lossHistory = {}
    local validationHistory = {}
    local ctcCriterion = nn.CTCCriterion()
    local x, gradParameters = self.model:getParameters()

    -- inputs (preallocate)
    local inputs = torch.Tensor()
    if self.nGPU > 0 then
        ctcCriterion = nn.CTCCriterion():cuda()
        inputs = inputs:cuda()
    end

    -- def loading buf and loader
    local loader = loader(self.lmdb_path)
    local spect_buf, label_buf

    -- load first batch
    local inds = self.indexer:nxt_inds()
    self.pool:addjob(function()
                    return loader:nxt_batch(inds, false)
                end,
                function(spect,label)
                    spect_buf=spect
                    label_buf=label
                end
                )

    -- ===========================================================
    -- define the feval
    -- ===========================================================
    local function feval(x_new)
        --------------------- data load ------------------------
        self.pool:synchronize()                         -- wait previous loading
        local inputsCPU,targets = spect_buf,label_buf   -- move buf to training data
        inds = self.indexer:nxt_inds()                  -- load nxt batch
        self.pool:addjob(function()
                            return loader:nxt_batch(inds, false)
                        end,
                        function(spect,label)
                            spect_buf=spect
                            label_buf=label
                        end
                        )

        --------------------- fwd and bwd ---------------------
        inputs:resize(inputsCPU:size()):copy(inputsCPU) -- transfer over to GPU
        gradParameters:zero()
        cutorch.synchronize()
        local predictions = self.model:forward(inputs)
        local loss = ctcCriterion:forward(predictions, targets)
        self.model:zeroGradParameters()
        local gradOutput = ctcCriterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        cutorch.synchronize()
        return loss, gradParameters
    end

    -- ==========================================================
    -- training
    -- ==========================================================
    local currentLoss
    local startTime = os.time()
    local dataSetSize = 20 -- TODO dataset:size()
    for i = 1, epochs do
        local averageLoss = 0
        print(string.format("Training Epoch: %d", i))

        local wer = WERValidationSet(self, validationDataset)
        if wer then table.insert(validationHistory, wer) end

        for j = 1, dataSetSize do
            currentLoss = 0
            local _, fs = optim.sgd(feval, x, sgd_params)
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, dataSetSize)
            averageLoss = averageLoss + currentLoss
        end

        averageLoss = averageLoss / dataSetSize -- Calculate the average loss at this epoch.
        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        print(string.format("Training Epoch: %d Average Loss: %f", i, averageLoss))

        logger:add{averageLoss, wer}
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
    saveDataParallel(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName)
    self.model = loadDataParallel(saveName, self.nGPU)
    model:evaluate()
end

return Network
