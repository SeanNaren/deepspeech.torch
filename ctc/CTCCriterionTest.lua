-- Tests are modelled around the results obtained in the examples here:
-- https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
require 'cunn'
require 'optim'
require 'rnn'
local CTCCriterion = require 'CTCCriterion'
function smallTest()
    local acts = {torch.Tensor({{0,0,0,0,0}}):float()}
    local targets = {{1}}
    assertion(CTCCriterion:updateOutput(acts,targets), 1.6094379425049, "CTCCriterion.smallTest")
end

function mediumTest()
    local acts = {torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}}):float()}
    local targets = {{3,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 7.355742931366, "CTCCriterion.mediumTest")
end

function mediumNegativeTest()
    local acts = {torch.Tensor({{-5,-4,-3,-2,-1},{-10,-9,-8,-7,-6},{-15,-14,-13,-12,-11}}):float()}
    local targets = {{2,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 4.9388499259949, "CTCCriterion.mediumNegativeTest")
end

-- The batch test takes the three above tests and combines them into one batch.
-- because of the CTCBatcher we have to pad the tensor of {{0,0,0,0}} to the max length tensor in the batch.
function batchTest()
    local acts = {
        torch.Tensor({{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}):float(),
        torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}}):float(),
        torch.Tensor({{-5,-4,-3,-2,-1},{-10,-9,-8,-7,-6},{-15,-14,-13,-12,-11}}):float()
    }
    local targets = {{1},{3,3},{2,3}}
    assertion(CTCCriterion:updateOutput(acts,targets), 5.1103823979696, "CTCCriterion.batchTest")
end

function assertion(outputCost,expectedCost,testName)
    if (tostring(outputCost) == tostring(expectedCost)) then
        print(testName," success")
    else
        print(testName," failed")
    end
end

smallTest()
mediumTest()
mediumNegativeTest()
batchTest()
