require 'loader'
require 'util'
local threads = require 'threads'
local Evaluator = require 'Evaluator'

local WERCalculator = {}

local _loader
function WERCalculator:init(_dir)
    _loader = loader(_dir)
    self.indexer = indexer(_dir, 1)
    self.pool = threads.Threads(1,function()require 'loader'end)
    local inds = self.indexer:nxt_inds()
    self.pool:addjob(function()
                        return _loader:nxt_batch(inds, false) -- set true to load trans
                    end,
                    function(spect, label, sizes)
                        self.spect_buf=spect
                        self.label_buf=label
                        self.sizes_buf=sizes
                    end
                    )
end

function WERCalculator:calculateValidationWER(gpu, model, calSize)

    -- Run sample of test data set through the net and print the results
    local sampleSize = 10
    local cumWER = 0
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if (gpu) then
        inputs = inputs:cuda()
    end

    for i = 1, sampleSize do
        self.pool:synchronize()
        local inputsCPU, targets, sizes = self.spect_buf, self.label_buf, self.sizes_buf
        local inds = self.indexer:nxt_inds()
        self.pool:addjob(function()
                        return _loader:nxt_batch(inds, false)
                        end,
                        function(spect, label, sizes)
                            self.spect_buf=spect
                            self.label_buf=label
                            self.sizes_buf=sizes
                        end
                        )
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        sizes = calSize(sizes)
        local prediction = model:forward({inputs, sizes})

        local predictedCharacters = Evaluator.getPredictedCharacters(prediction)
        local WER = Evaluator.sequenceErrorRate(targets, predictedCharacters)

        cumWER = cumWER + WER
    end

    return (cumWER / sampleSize)
end

return WERCalculator
