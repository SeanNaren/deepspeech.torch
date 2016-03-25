--[[
-- VERY basic spell checker based on Peter Norvig's spell correct model.
-- http://norvig.com/spell-correct.html
]]

require 'lfs'

local SpellingChecker = {}

local words = {}

local alphabet = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
}

function SpellingChecker:init(wordsFilePath)
    words = {}
    for line in io.lines(wordsFilePath) do
        line = string.lower(line)
        for word in string.gmatch(line, '%w+') do
            local count = words[word]
            if (count ~= nil) then
                words[word] = words[word] + 1
            else
                words[word] = 1
            end
        end
    end
end

-- Input is a word in string format.
-- Returns a list of edits to the words based on deleting/transposing/replacing/inserting characters.
local function edits(word)
    local results = {}
    for i = 1, word:len() do
        local deletion = string.sub(word, 1, i - 1) .. string.sub(word, i + 1)
        table.insert(results, deletion)
    end
    for i = 1, word:len() - 1 do
        local tranpose = string.sub(word, 1, i - 1) .. string.sub(word, i + 1, i + 1) .. string.sub(word, i, i) .. string.sub(word, i + 2)
        table.insert(results, tranpose)
    end
    for i = 1, word:len() do
        for index, c in ipairs(alphabet) do
            local replacement = string.sub(word, 1, i - 1) .. c .. string.sub(word, i + 1)
            table.insert(results, replacement)
        end
    end
    for i = 1, word:len() + 1 do
        for index, c in ipairs(alphabet) do
            local insertion = string.sub(word, 1, i - 1) .. c .. string.sub(word, i)
            table.insert(results, insertion)
        end
    end
    return results
end

local function maxKey(table)
    local key = next(table)
    local max = table[key]
    for k, v in pairs(table) do
        if table[k] > max then
            key, max = k, v
        end
    end
    return key
end

function SpellingChecker:correct(word)
    if (words[word] ~= nil) then
        return word
    else
        local list = edits(word)
        local candidates = {}
        local numberOfCandidates = 0
        for index, s in ipairs(list) do
            if (words[s] ~= nil) then
                candidates[words[s]] = s
                numberOfCandidates = numberOfCandidates + 1
            end
        end
        if (numberOfCandidates > 0) then
            return candidates[maxKey(candidates)]
        end
        -- Repeat again for error distance of 2.
        for index, s in ipairs(list) do
            for index, w in ipairs(edits(s)) do
                if (words[w] ~= nil) then
                    candidates[words[w]] = w
                    numberOfCandidates = numberOfCandidates + 1
                end
            end
        end
        if (numberOfCandidates > 0) then
            return candidates[maxKey(candidates)]
        end
        return word
    end
end

return SpellingChecker

