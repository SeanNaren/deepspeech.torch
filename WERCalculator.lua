local Evaluator = require 'Evaluator'

local WERCalculator = {}
function WERCalculator.calculateValidationWER(testDataSet, gpu, model)

    -- Run sample of test data set through the net and print the results
    local sampleSize = 10
    local sample = torch.rand(sampleSize):mul(#testDataSet):add(0.5):round()
    local cumWER = 0
    local input = torch.Tensor()
    if (gpu == true) then input = input:cuda() end

    for i = 1, sample:size(1) do
        local inputCPU = testDataSet[sample[i]].input
        local targets = testDataSet[sample[i]].label
        -- transfer over to GPU
        input:resize(1, 1, inputCPU:size(1), inputCPU:size(2))
        input[1][1]:copy(inputCPU)
        local prediction = model:forward(input)

        local predictedCharacters = Evaluator.getPredictedCharacters(prediction)
        local WER = Evaluator.sequenceErrorRate(targets, predictedCharacters)

        cumWER = cumWER + WER
    end

    return (cumWER / sampleSize)
end

return WERCalculator