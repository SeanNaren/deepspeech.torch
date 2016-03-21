-- Tests are modelled around the results obtained in the examples here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
require 'cunn'
require 'optim'
require 'rnn'


local CTCCriterion = require 'CTCCriterion'

function smallTest()
    local acts = {torch.Tensor({0,0,0,0,0}):cuda()}
    local targets = {{1}}
    assertion(CTCCriterion:updateOutput(acts,targets), 1.6094379425049, "CTCCriterion.smallTest")
end

function mediumTest()
    local acts = {
        torch.Tensor({1,2,3,4,5}):cuda(),
        torch.Tensor({6,7,8,9,10}):cuda(),
        torch.Tensor({11,12,13,14,15}):cuda()
    }
    local targets = {{3,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 7.355742931366, "CTCCriterion.mediumTest")
end

function mediumNegativeTest()
    local acts = {
        torch.Tensor({-5,-4,-3,-2,-1}):cuda(),
        torch.Tensor({-10,-9,-8,-7,-6}):cuda(),
        torch.Tensor({-15,-14,-13,-12,-11}):cuda()
    }
    local targets = {{2,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 4.938850402832, "CTCCriterion.mediumNegativeTest")
end

-- TODO batching is currently not supported, hopefully soon.
-- The batch test takes the three above tests and combines them into one batch.
-- because of the CTCBatcher we have to pad the tensor of {{0,0,0,0}} to the max length tensor in the batch.
function batchTest()
--    local acts = {
--        torch.Tensor({{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}):float(),
--        torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}}):float(),
--        torch.Tensor({{-5,-4,-3,-2,-1},{-10,-9,-8,-7,-6},{-15,-14,-13,-12,-11}}):float()
--    }
--    local targets = {{1},{3,3},{2,3}}
--    assertion(CTCCriterion:updateOutput(acts,targets), 5.1103823979696, "CTCCriterion.batchTest")
end

function assertion(outputCost,expectedCost,testName)
    if (tostring(outputCost) == tostring(expectedCost)) then
        print(testName,"success")
    else
        print(testName,"failed")
    end
end

-- Tests are modelled around examples given in the CTCBatcher class

function convertToCTCBatchSequenceTest()
    local input = {torch.Tensor({1,2,3,4,5}),torch.Tensor({6,7,8,9,10})}
    local output = CTCCriterion.convertToCTCBatchSequence(input)
    local expected = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}})
    assertion(output, expected, "convertToCTCBatchSequenceTest")
end

function convertToNetSequenceTest()
    local input = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}})
    local numberOfSequences = 2
    local output = CTCCriterion.convertToNetSequence(input,numberOfSequences)
    local expected = {torch.Tensor({1,2,3,4,5}):cuda(),torch.Tensor({6,7,8,9,10}):cuda()}
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

smallTest()
mediumTest()
mediumNegativeTest()
batchTest()

--CTC batching tests.
convertToCTCBatchSequenceTest()
convertToNetSequenceTest()