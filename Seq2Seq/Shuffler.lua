-- Takes the original sentences and shuffles characters around to simulate 'errors'.

local Shuffler = {}

math.randomseed(os.time())

function Shuffler.shuffleSentences(originalSentences, randomness)
    local shuffledSentences = {}
    for index, sentence in ipairs(originalSentences) do
        --Some letters are missing
        local chosenShuffle = math.random(3)
        if (chosenShuffle == 1) then
            for x = 1, math.random(randomness) do
                sentence[math.random(#sentence)] = nil
                local temp = {}
                for i = 1, #sentence do
                    if (sentence[i] ~= nil) then
                        table.insert(temp, sentence[i])
                    end
                end
                sentence = temp
            end
        elseif (chosenShuffle == 2) then
            --Some added labels (randomly between 1-26)
            for x = 1, math.random(randomness) do
                table.insert(sentence, math.random(#sentence), math.random(26))
            end
        elseif (chosenShuffle == 3) then
            --Swap letters with adjacent.
            for x = 1, math.random(randomness) do
                local swapIndex = math.random(2, #sentence)
                sentence[swapIndex], sentence[swapIndex - 1] = sentence[swapIndex - 1], sentence[swapIndex]
                table.insert(sentence, math.random(#sentence), math.random(26))
            end
        end
        table.insert(shuffledSentences, sentence)
    end
    return shuffledSentences
end

-- TODO this should not be here, temporarily here to evaluate success of spell checker visually.
local alphabet = {
    '$', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' '
}

local alphabetMapping = {}
local indexMapping = {}
for index, character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

--Given an index returns the letter at that index.
function Shuffler.findLetter(index)
    return indexMapping[index]
end

return Shuffler