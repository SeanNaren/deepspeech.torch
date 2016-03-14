--Aids in the batching process of sequences into the Warp-ctc format
--described here at the end of this README:
--https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md

--Converts the predictions made by the network (activation sequence) to the
--batch format. ALL SEQUENCES MUST BE THE SAME LENGTH.
--(padding has to be included if any irregular sequences)
--Tensors - The sequence tensors obtained from the forwarding of inputs through the network.
--Example:
--input to function: {torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{0,0,0,0,0}})}
--Returned batched format: torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
function convertToCTCBatchSequence(tensors)
    local columnMajor = {}
        for index, tensor in ipairs(tensors) do
            table.insert(columnMajor, getTensorValue(tensor))
        end
    local resultTensor = torch.Tensor(columnMajor)
    return resultTensor
end

--Reverses the batching process to give the gradientOutput for
--backwards propagation on the net.
--Example:
--input to function (output of the CTCCriterion):
--torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
--Returned format:
--{torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{0,0,0,0,0}})}
function convertToNetSequence(gradientOutput, numberOfSequences)
    local gradients = {}
    for i = 1, numberOfSequences do
        table.insert(gradients, {})
    end
    for i = 1, gradientOutput:size(1) do
        local index = math.fmod(i, numberOfSequences)
        if (index == 0) then index = numberOfSequences end
        table.insert(gradients[index], torch.totable(gradientOutput[i]))
    end
    local returnTensors = {}
    for i = 1, numberOfSequences do
        table.insert(returnTensors, torch.CudaTensor(gradients[i]):squeeze())
    end
    return returnTensors
end

--Gets the tensor's value in the sequence.
function getTensorValue(tensor)
    local tableTensorValue = torch.totable(tensor)
    return tableTensorValue
end
