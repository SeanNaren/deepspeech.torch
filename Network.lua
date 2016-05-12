require 'optim'
require 'nnx'
require 'BRNN'
require 'ctchelpers'
require 'gnuplot'
require 'xlua'
require 'utils_multi_gpu'
require 'loader'
require 'mapper'
require 'wer_tester'

local threads = require 'threads'
local Network = {}
local logger = optim.Logger('train.log')
logger:setNames { 'loss'}
logger:style { '-'}


function Network:init(networkParams)
    
    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.nGPU = networkParams.nGPU
    self.lmdb_path = networkParams.lmdb_path
    self.val_path = networkParams.val_path
    self.mapper = mapper(networkParams.dict_path)
    self.wer_tester = wer_tester(self.val_path, self.mapper, networkParams.test_batch_size, 
        networkParams.test_iter)

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
    self.indexer:prep_same_len_inds() -- rm this if zero-masking is done
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


local function test(self)
    print('testing...')
    self.model:evaluate()
    local wer = self.wer_tester:get_wer(self.nGPU>0, self.model, true) -- detail in log 
    self.model:zeroGradParameters()
    self.model:training()
    return wer
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
    local spect_buf, label_buf, sizes_buf

    -- load first batch
    local inds = self.indexer:nxt_same_len_inds() -- use nxt_inds if zero-mask is done
    self.pool:addjob(function()
                        return loader:nxt_batch(inds, false)
                    end,
                    function(spect,label,sizes)
                        spect_buf=spect
                        label_buf=label
                        sizes_buf=sizes
                    end
                    )

    -- ===========================================================
    -- define the feval
    -- ===========================================================
    local function feval(x_new)
        --------------------- data load ------------------------
        self.pool:synchronize()                         -- wait previous loading
        local inputsCPU,targets = spect_buf,label_buf   -- move buf to training data
        inds = self.indexer:nxt_same_len_inds()                  -- load nxt batch
        self.pool:addjob(function()
                            return loader:nxt_batch(inds, false)
                        end,
                        function(spect,label,sizes)
                            spect_buf=spect
                            label_buf=label
                            sizes_buf=sizes
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
    local dataSetSize = self.indexer.len_num -- obtained when calling prep_same_len_inds

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
        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        print(string.format("Training Epoch: %d Average Loss: %f", i, averageLoss))
        
        -- Periodically update validation error rates
        if (i % 2 == 0 and  self.val_path) then
            local wer = test(self)
            if wer then
                table.insert(validationHistory, 100 * wer)
                print('Training Epoch: '..i..' averaged WER: '.. 100*wer ..'%')
            end
        end

        logger:add { averageLoss}
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
