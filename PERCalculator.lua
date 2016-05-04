local Evaluator = require 'Evaluator'

local PERCalculator = {}
function PERCalculator.calculateValidationPER(testDataSet, gpu, model)

    -- Run sample of test data set through the net and print the results
    local sampleSize = 10
    local sample = torch.rand(sampleSize):mul(#testDataSet):add(0.5):round()
    local cumPER = 0
    local input = torch.Tensor()
    if (gpu == true) then input = input:cuda() end

    for i = 1, sample:size(1) do
        local inputCPU = testDataSet[sample[i]].input
        local targets = testDataSet[sample[i]].labels
        -- transfer over to GPU
        input:resize(1, 1, inputCPU:size(1), inputCPU:size(2))
        input[1][1]:copy(inputCPU)
        local prediction = model:forward(input)

        local predictedPhones = Evaluator.getPredictedPhones(prediction)
        local PER = Evaluator.sequenceErrorRate(targets, predictedPhones)

        cumPER = cumPER + PER
    end

    return (cumPER / sampleSize)
end

return PERCalculator