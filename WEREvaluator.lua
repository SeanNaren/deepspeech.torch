require 'Loader'
require 'Utils'
require 'Mapper'
require 'torch'
require 'xlua'
local threads = require 'threads'
local Evaluator = require 'Evaluator'

local WEREvaluator = torch.class('WEREvaluator')

local _loader

function WEREvaluator:__init(datasetPath, mapper, testBatchSize, nbOfTestIterations, logsPath)
    _loader = Loader(datasetPath)
    self.testBatchSize = testBatchSize
    self.nbOfTestIterations = nbOfTestIterations
    self.indexer = indexer(datasetPath, testBatchSize)
    self.pool = threads.Threads(1, function() require 'Loader' end)
    self.mapper = mapper
    self.logsPath = logsPath
    self.suffix = '_' .. os.date('%Y%m%d_%H%M%S')
end

function WEREvaluator:getWER(gpu, model, verbose, epoch)
    --[[
        load test_iter*batch_size data point from test set; compute average WER

        input:
            verbose:if true then print WER and predicted strings for each data to log
    --]]

    local cumWER = 0
    local inputs = torch.Tensor()
    if (gpu) then
        inputs = inputs:cuda()
    end
    local spect_buf, label_buf, sizes_buf

    -- get first batch
    local inds = self.indexer:nxt_inds()
    self.pool:addjob(function()
        return _loader:nxt_batch(inds, false)
    end,
        function(spect, label, sizes)
            spect_buf = spect
            label_buf = label
            sizes_buf = sizes
        end)

    if verbose then
        local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'), "Could not create validation test logs, does the folder "
                .. self.logsPath .. " exist?")
        f:write('======================== BEGIN WER TEST EPOCH: ' .. epoch .. ' =========================\n')
        f:close()
    end

    local werPredictions = {} -- stores the predictions to order for log.

    -- ======================= for every test iteration ==========================
    for i = 1, self.nbOfTestIterations do
        -- get buf and fetch next one
        self.pool:synchronize()
        local inputsCPU, targets, sizes_array = spect_buf, label_buf, sizes_buf
        inds = self.indexer:nxt_inds()
        self.pool:addjob(function()
            return _loader:nxt_batch(inds, true)
        end,
            function(spect, label, sizes)
                spect_buf = spect
                label_buf = label
                sizes_buf = sizes
            end)

        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        if(gpu) then cutorch.synchronize() end
        local predictions = model:forward(inputs)
        predictions = predictions:view(-1, self.testBatchSize, predictions:size(2)):transpose(1, 2)
        if(gpu) then cutorch.synchronize() end

        -- =============== for every data point in this batch ==================
        for j = 1, self.testBatchSize do
            local prediction_single = predictions[j]
            local predict_tokens = Evaluator.predict2tokens(prediction_single, self.mapper)
            local WER = Evaluator.sequenceErrorRate(targets[j], predict_tokens)
            cumWER = cumWER + WER
            table.insert(werPredictions, { wer = WER * 100, target = self:tokens2text(targets[j]), prediction = self:tokens2text(predict_tokens) })
        end
    end

    local function comp(a, b) return a.wer < b.wer end

    table.sort(werPredictions, comp)

    if verbose then
        for index, werPrediction in ipairs(werPredictions) do
            local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'))
            f:write(string.format("WER = %.2f%% | Text = \"%s\" | Predict = \"%s\"\n",
                werPrediction.wer, werPrediction.target, werPrediction.prediction))
            f:close()
        end
    end
    local averageWER = cumWER / (self.nbOfTestIterations * self.testBatchSize)
    local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'))
    f:write(string.format("Average WER = %.2f%%", averageWER * 100))
    f:close()

    self.pool:synchronize() -- end the last loading
    return averageWER
end

function WEREvaluator:tokens2text(tokens)
    local text = ""
    for i, t in ipairs(tokens) do
        text = text .. self.mapper.token2alphabet[tokens[i]]
    end
    return text
end
