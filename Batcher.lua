require 'nn'
local Batcher = {}

function Batcher.createMinibatchDataset(dataSet, maxSizeDiff)

    -- Sort on the time dimension, which dim(2)
    table.sort(dataSet, function (rowA, rowB) if (rowA.input:size(2) < rowB.input:size(2)) then return true else return false end end)

    local miniBatchInputs = {}
    local miniBatchLabels = {}
    local rowNum = 1
    local currentMiniBatch = {}
    local refRow = dataSet[1]

    local function createBatchTensor()
        -- The largest input governs the size of this batch.
        local biggestInputSize = currentMiniBatch[#currentMiniBatch].input:size()

        -- The network uses spatial convolution which requires turning (txf) into a an image layer
        -- so we add a wrapper dimenion to achieve this
        local batchInputs = torch.Tensor(#currentMiniBatch, 1, biggestInputSize[1], biggestInputSize[2])
        local batchLabels = {}

        for index, row in ipairs(currentMiniBatch) do

            local temp = torch.zeros(biggestInputSize[1], biggestInputSize[2])
            temp:narrow(2, 1, row.input:size(2)):copy(row.input)
            -- We add 1 dimension (1 feature map).
            batchInputs[index] = temp:view(1, biggestInputSize[1], biggestInputSize[2])

            table.insert(batchLabels, row.labels)
        end
        return batchInputs, batchLabels
    end

    -- This is the main loop for creating minibatches which are in order of input size
    -- This makes life easier for training: start with easy inputs and work up through harder ones
    while (rowNum <= #dataSet) do
        -- Sweep up rows that are within maxSizeDiff of each other
        if ((dataSet[rowNum].input:size(2) <= refRow.input:size(2) + maxSizeDiff))  then
            table.insert(currentMiniBatch, dataSet[rowNum])
        end

        -- If we have exceeded the max allowed size difference or we're at the last row
        -- we finish the current minibatch and append it to the batch.
        if((dataSet[rowNum].input:size(2) > refRow.input:size(2) + maxSizeDiff) or
                rowNum == #dataSet) then
            -- Finish current minibatch
            local batchInputs, batchLabels = createBatchTensor()
            table.insert(miniBatchInputs,  batchInputs)
            table.insert(miniBatchLabels, batchLabels)
            refRow = dataSet[rowNum]

            -- Start new minibatch (moot if we're at the last row)
            currentMiniBatch = {}
            table.insert(currentMiniBatch, refRow)
        end

        rowNum = rowNum + 1
    end

    -- Return a wrapper class/object to the minibatches
    return createDataSet(miniBatchInputs, miniBatchLabels)
end

-- A class wrapper that implements "size()" and "nextData()" for the minibatches created above
function createDataSet(miniBatchInputs, miniBatchLabels)
    local dataset = {}
    local pointer = 1

    -- Public interface
    function dataset:size() return #miniBatchInputs end

    -- Public interface
    function dataset:nextData()
        pointer = pointer + 1
        if (pointer > dataset:size()) then pointer = 1 end
        return miniBatchInputs[pointer], miniBatchLabels[pointer]
    end

    return dataset
end

return Batcher