require 'CTCBatcher'
-- Tests are modelled around examples given in the CTCBatcher class

function convertToCTCBatchSequenceTest()
    local input = {torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{0,0,0,0,0}})}
    local output = convertToCTCBatchSequence(input)
    local expected = torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
    assertion(output, expected, "convertToCTCBatchSequenceTest")
end

function convertToNetSequenceTest()
    local input = torch.Tensor({{1,2,3,4,5},{11,12,13,14,15},{6,7,8,9,10},{0,0,0,0,0}})
    local numberOfSequences = 2
    local output = convertToNetSequence(input,numberOfSequences)
    local expected = {torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}),torch.Tensor({{11,12,13,14,15},{0,0,0,0,0}})}
    assertion(output[1], expected[1], "convertToNetSequenceTest 1st output")
    assertion(output[2], expected[2], "convertToNetSequenceTest 2nd output")
end

function assertion(output, expected,testName)
    if (tostring(output) == tostring(expected)) then
        print(testName, "success")
    else
        print(testName, "failed")
    end
end

convertToCTCBatchSequenceTest()
convertToNetSequenceTest()