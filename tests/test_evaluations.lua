require 'nn'

local test = torch.TestSuite()
local mytester
local evaluator = require '../Evaluator'
require '../Mapper'

function test.evaluator()
    -- Calculates WER, (nbOfInsertions + nbOfDeletions + nbOfSubstitutions) / nbOfWords
    local target = { "test", "a", "sentence" }

    local prediction = { "a", "sentence" }
    local deletion = evaluator.sequenceErrorRate(target, prediction)
    local prediction = { "test", "a", "sentence", "inserted" }
    local insertion = evaluator.sequenceErrorRate(target, prediction)
    local prediction = { "test", "substituted", "sentence" }
    local substitution = evaluator.sequenceErrorRate(target, prediction)
    local oneMistakeWER = 1 / 3 -- One insertion/deletion/substitution / number of words
    mytester:eq(deletion, oneMistakeWER, 'WER with deletion was incorrect')
    mytester:eq(insertion, oneMistakeWER, 'WER with insertion was incorrect')
    mytester:eq(substitution, oneMistakeWER, 'WER with substitution was incorrect')

    local prediction = { "a" }
    local deletion = evaluator.sequenceErrorRate(target, prediction)
    local prediction = { "a", "wrong" }
    local deletionAndSubstitution = evaluator.sequenceErrorRate(target, prediction)
    local prediction = { "wrong", "a", "sentence", "inserted" }
    local substitionAndInsertion = evaluator.sequenceErrorRate(target, prediction)
    local twoMistakeWER = 2 / 3 -- Two errors of insertion/deletion/substitution / number of words
    mytester:eq(deletion, twoMistakeWER, 'masking of outputs was incorrect')
    mytester:eq(deletionAndSubstitution, twoMistakeWER, 'WER with substitution and deletion was incorrect')
    mytester:eq(substitionAndInsertion, twoMistakeWER, 'WER with substitution and insertion was incorrect')
end

function test.evaluatorPredict2tokens()
    local dir_path = '../dictionary'
    local mapper = Mapper(dir_path)
    local predictions = torch.Tensor({ { 1, 2, 3 }, { 2, 3, 1 }, { 1, 2, 3 } })
    local tokens = evaluator.predict2tokens(predictions, mapper)
    mytester:eq(tokens, { 2, 1, 2 })
end

function test.mapper()
    local dir_path = '../dictionary'
    local mapper = Mapper(dir_path)
    local alphabet = {
        '$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '
    }
    local expectedMapping = {}
    for index, letter in ipairs(alphabet) do
        expectedMapping[letter] = index - 1
    end
    mytester:eq(mapper.alphabet2token, expectedMapping)
end

mytester = torch.Tester()
mytester:add(test)
mytester:run()
