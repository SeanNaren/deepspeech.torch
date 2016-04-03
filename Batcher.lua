require 'nn'
local Batcher = {}
-- This class should create exactly what is needed by the CTC class.
-- We need the input tensor of each mini-batch to be a padded 3d tensor of batchsize x time x freq.
-- We need the labels to be {{1,3},{3,1}} etc for each sample in the batch.

-- The batched CTC activations and the size of tensors are handled by the CTCCriterion.

-- tensorsAndTargets is a table such that {{tensor = torch.Tensor(), label = {1,3}},etc}
function Batcher.createMinibatchDataset(tensorsAndTargets, maximumSizeDifference)

    local function sortFunction(tensorX, tensorY)
        if (tensorX.tensor:size(1) < tensorY.tensor:size(1)) then return true else return false end
    end

    table.sort(tensorsAndTargets, sortFunction)

    local miniBatches = {}
    local miniBatchesTarget = {}
    local counter = 1
    local batch = {}
    local currentTensor = tensorsAndTargets[1]

    local function createBatchTensor()
        local biggestTensor = batch[#batch].tensor:size()
        local batchTensor = torch.Tensor(#batch, 1, biggestTensor[1], biggestTensor[2]):transpose(3,4) -- We add 1 dimension (1 feature map).
        local batchTargets = {}
        for index, tensorAndTarget in ipairs(batch) do
            local output = torch.zeros(biggestTensor[1], biggestTensor[2])
            local area = output:narrow(1, 1, tensorAndTarget.tensor:size(1)):copy(tensorAndTarget.tensor)
            batchTensor[index] = output:view(1, biggestTensor[1], biggestTensor[2]):transpose(2,3) -- We add 1 dimension (1 feature map).
            table.insert(batchTargets, tensorAndTarget.label)
        end
        return batchTensor, batchTargets
    end

    while (counter <= #tensorsAndTargets) do
        if (tensorsAndTargets[counter].tensor:size(1) - maximumSizeDifference <= currentTensor.tensor:size(1)) then
            table.insert(batch, tensorsAndTargets[counter])
        else
            local batchTensor, batchTarget = createBatchTensor()
            table.insert(miniBatches, batchTensor)
            table.insert(miniBatchesTarget, batchTarget)
            currentTensor = tensorsAndTargets[counter]
            batch = {}
            table.insert(batch, currentTensor)
        end

        if (counter == #tensorsAndTargets and #batch ~= 0) then
            local batchTensor, batchTarget = createBatchTensor()
            table.insert(miniBatches, batchTensor)
            table.insert(miniBatchesTarget, batchTarget)
            currentTensor = tensorsAndTargets[counter]
            batch = {}
            table.insert(batch, currentTensor)
        end

        counter = counter + 1
    end
    local dataset = createDataSet(miniBatches, miniBatchesTarget)
    return dataset
end

function createDataSet(miniBatches, miniBatchesTarget)
    local dataset = {}
    local pointer = 1
    function dataset:size() return #miniBatches end

    function dataset:nextData()
        pointer = pointer + 1
        if (pointer > dataset:size()) then pointer = 1 end
        return miniBatches[pointer], miniBatchesTarget[pointer]
    end

    return dataset
end

return Batcher