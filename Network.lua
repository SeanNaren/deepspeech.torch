require 'optim'
require 'nnx'
require 'gnuplot'
require 'xlua'
require 'utils_multi_gpu'
require 'loader'
require 'nngraph'
require 'mapper'
require 'wer_tester'
require 'cudnn'

local suffix = '_'..os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}
local logger = optim.Logger('train'..suffix..'.log')
logger:setNames {'loss', 'WER'}
logger:style {'-', '-'}


function Network:init(networkParams)

    self.fileName = networkParams.fileName -- The file name to save/load the network from.
    self.nGPU = networkParams.nGPU
    if self.nGPU <= 0 then
        assert(networkParams.backend ~= 'cudnn')
    end
    assert(networkParams.batch_size % networkParams.nGPU == 0, 'batch size must be the multiple of nGPU')
    self.gpu_batch_size = networkParams.batch_size / networkParams.nGPU

    self.lmdb_path = networkParams.lmdb_path
    self.val_path = networkParams.val_path
    self.mapper = mapper(networkParams.dict_path)
    self.wer_tester = wer_tester(self.val_path, self.mapper, networkParams.test_batch_size,
        networkParams.test_iter)
    self.saveModel = networkParams.saveModel
    self.loadModel = networkParams.loadModel
    self.snap_shot_epochs = networkParams.snap_shot_epochs or 10

    -- setting model saving/loading
    if (self.loadModel) then
        assert(networkParams.fileName, "Filename hasn't been given to load model.")
        self:loadNetwork(networkParams.fileName,
            networkParams.modelName,
            networkParams.backend == 'cudnn')
    else
        assert(networkParams.modelName, "Must have given a model to train.")
        self:prepSpeechModel(networkParams.modelName, networkParams.backend)
    end
    local typename = torch.typename(self.model)
    local print_model = self.model
    if typename == 'nn.DataParallelTable' then
        print_model = self.model:get(1)
        typename = torch.typename(print_model)
    end
    if typename == 'nn.gModule' then
        graph.dot(print_model.fg, networkParams.modelName, networkParams.modelName) -- view graph
    else
        print (print_model)
    end
    assert((networkParams.saveModel or networkParams.loadModel) and networkParams.fileName, "To save/load you must specify the fileName you want to save to")
    -- setting online loading
    self.indexer = indexer(networkParams.lmdb_path, networkParams.batch_size)
    self.indexer:prep_sorted_inds()
    self.pool = threads.Threads(1,function() require 'loader' end)
    self.batch_num = math.floor(self.indexer.lmdb_size / networkParams.batch_size)
end


function Network:prepSpeechModel(modelName, backend)
    local model = require (modelName)
    self.model = model[1](self.nGPU, backend=='cudnn')
    self.calSize = model[2]
end


function Network:testNetwork()
    print('testing...')
    self.model:evaluate()
    local wer = self.wer_tester:get_wer(self.nGPU>0, cudnn.convert(self.model, nn), self.calSize, true) -- detail in log
    self.model:zeroGradParameters()
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

    -- inputs (preallocate)
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if self.nGPU > 0 then
        inputs = inputs:cuda()
        sizes = sizes:cuda()
    end

    -- def loading buf and loader
    local loader = loader(self.lmdb_path)
    local spect_buf, label_buf, sizes_buf

    -- load first batch
    local inds = self.indexer:nxt_sorted_inds()
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
        local inputsCPU,sizes,targets = spect_buf,sizes_buf,label_buf   -- move buf to training data
        inds = self.indexer:nxt_sorted_inds()                  -- load nxt batch
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
        sizes = self.calSize(sizes)
        self.model:forward({inputs, sizes})
        self.model:zeroGradParameters()
        local loss = self.model:backward(inputs, targets)
        gradParameters:div(inputs:size(1))

        return loss, gradParameters
    end

    -- ==========================================================
    -- training
    -- ==========================================================
    local currentLoss
    local startTime = os.time()
    -- local dataSetSize = self.indexer.len_num -- obtained when calling prep_same_len_inds

    for i = 1, epochs do
        local averageLoss = 0
        for j = 1, self.batch_num do
            currentLoss = 0
            cutorch.synchronize()
            local _, fs = optim.sgd(feval, x, sgd_params)
            cutorch.synchronize()
            if self.model.needsSync then
                self.model:syncParameters()
            end
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.batch_num)
            averageLoss = averageLoss + currentLoss
        end
        averageLoss = averageLoss / self.batch_num -- Calculate the average loss at this epoch.
        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        print(string.format("Training Epoch: %d Average Loss: %f", i, averageLoss))

        -- Periodically update validation error rates
        local wer = self:testNetwork()
--        table.insert(validationHistory, 100 * wer)
       print('Training Epoch: '.. i ..' averaged WER: '.. 100*wer ..'%')
--        logger:add {averageLoss, wer}

        -- periodically save the model
        if self.saveModel and i % self.snap_shot_epochs == 0 then
            print("Saving model..")
            self:saveNetwork('epoch_'..i..suffix..self.fileName)
        end

    end

    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)

    return lossHistory, validationHistory, minutesTaken
end


function Network:createLossGraph()
    logger:plot()
end


function Network:saveNetwork(saveName)
    saveDataParallel(saveName, self.model)
end


--Loads the model into Network.
function Network:loadNetwork(saveName, modelName, is_cudnn)
    self.model = loadDataParallel(saveName, self.nGPU, is_cudnn)
    local model = require (modelName)
    self.calSize = model[2]
end

return Network
