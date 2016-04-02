-- Tests are modelled around the results obtained in the examples here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
require 'cunn'
require 'optim'
require 'rnn'
require 'CTCCriterion'

local CTCCriterion = nn.CTCCriterion():cuda()

function smallTest()
    local acts = torch.Tensor({{{0,0,0,0,0}}}):cuda()
    local targets = {{1}}
    assertion(CTCCriterion:updateOutput(acts,targets), 1.6094379425049, "CTCCriterion.smallTest")
end

function mediumTest()
    local acts =
        torch.Tensor({
            {{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}}
        }):cuda()

    local targets = {{3,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 7.355742931366, "CTCCriterion.mediumTest")
end

function mediumNegativeTest()
    local acts =
        torch.Tensor({
            {{-5,-4,-3,-2,-1}, {-10,-9,-8,-7,-6}, {-15,-14,-13,-12,-11}}
            }):cuda()

    local targets = {{2,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 4.938850402832, "CTCCriterion.mediumNegativeTest")
end


-- The batch test takes the three above tests and combines them into one batch.
-- The batch sequences must be of the same length (length of 3 below) (as a result the test results are slightly
-- different to the Torch CTC binding tutorial README due to the first column also having 3 elements rather than 1.)
function batchTest()
    local acts =
        torch.Tensor({
            {{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}},
            {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},
            {{-5,-4,-3,-2,-1},{-10,-9,-8,-7,-6},{-15,-14,-13,-12,-11}}
            }):cuda()

    local targets = {{1},{3,3},{2,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 15.331147670746, "CTCCriterion.batchTest")
end

function assertion(outputCost,expectedCost,testName)
    if (tostring(outputCost) == tostring(expectedCost)) then
        print(testName,"success")
    else
        print(testName,"failed")
    end
end

-- Takes a 3D tensor of batch x time x freq and converts this into the 2d batch tensor form.
function createCTCBatchTest()
    local input = torch.Tensor({{{1,2,3,4,5}},{{6,7,8,9,10}}})
    local output = CTCCriterion:createCTCBatch(input, input:size())
    local expected = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}}):cuda()
    assertion(output, expected, "createCTCBatchTest")
end

-- Reverts the 2d batch tensor form from CTC into a 3d tensor of batch x time x freq
function revertBatchingTest()
    local input = torch.Tensor({{{1,2,3,4,5}},{{6,7,8,9,10}}})
    local gradient = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10}})

    local output = CTCCriterion:revertBatching(gradient, input:size())

    local expected = torch.Tensor({{{1,2,3,4,5}},{{6,7,8,9,10}}}):cuda()
    assertion(output[1], expected[1], "revertBatchingTest 1st output")
    assertion(output[2], expected[2], "revertBatchingTest 2nd output")
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

--CTC sequencing tests.
createCTCBatchTest()
revertBatchingTest()