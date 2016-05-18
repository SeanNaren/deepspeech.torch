require 'loader'
require 'util'
require 'mapper'
require 'torch'
require 'xlua'
require 'cutorch'
local threads = require 'threads'
local Evaluator = require 'Evaluator'

-- register this class
local wer_tester = torch.class('wer_tester')

local _loader -- ugly solution



function wer_tester:__init(_path, mapper, test_batch_size, test_iter)
    _loader = loader(_path)
    self.test_batch_size = test_batch_size
    self.test_iter = test_iter
    self.indexer = indexer(_path, test_batch_size)
    self.pool = threads.Threads(1,function()require 'loader'end)
    self.mapper = mapper
    self.suffix = '_'..os.date('%Y%m%d_%H%M%S')
end



function wer_tester:get_wer(gpu, model, calSize, verbose)
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
                        spect_buf=spect
                        label_buf=label
                        sizes_buf=sizes
                    end
                    )

    if verbose then
        local f = assert(io.open('test'..self.suffix..'.log', 'a'))
        f:write('======================== begin test =========================\n')
        f:close()
    end

    -- ======================= for every test iteration ==========================
    for i = 1, self.test_iter do

        -- get buf and fetch next one
        self.pool:synchronize()
        local inputsCPU, targets, sizes_array = spect_buf, label_buf, sizes_buf
        inds = self.indexer:nxt_inds()
        self.pool:addjob(function()
                            return _loader:nxt_batch(inds, true)
                        end,
                        function(spect, label, sizes)
                            spect_buf=spect
                            label_buf=label
                            sizes_buf=sizes
                        end
                        )

        sizes_array = calSize(sizes_array)
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        cutorch.synchronize()
        local predictions = model:forward({inputs,sizes_array})
        predictions = predictions:view(self.test_batch_size, -1, predictions:size(2)):transpose(1, 2)
        cutorch.synchronize()

        -- =============== for every data point in this batch ==================
        for j = 1, self.test_batch_size do

            local prediction_single = predictions[j]
            local predict_tokens = Evaluator.predict2tokens(prediction_single, self.mapper)
            local WER = Evaluator.sequenceErrorRate(targets[j], predict_tokens)
            cumWER = cumWER + WER

            if verbose then
                local f = assert(io.open('test'..self.suffix..'.log', 'a'))
                f:write(string.format("WER = %.0f%% | Text = \"%s\" | Predict = \"%s\"\n",
                    WER*100, self:tokens2text(targets[j]), self:tokens2text(predict_tokens)))
                f:close()
            end

        end
        xlua.progress(i, self.test_iter)
    end

    self.pool:synchronize() -- end the last loading
    return cumWER / (self.test_iter*self.test_batch_size)
end



function wer_tester:tokens2text(tokens)
    local text = ""
    for i,t in ipairs(tokens) do
        text = text .. self.mapper.token2alphabet[tokens[i]]
    end
    return text
end
