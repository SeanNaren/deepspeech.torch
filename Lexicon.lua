-- [Do I need this to be fixed?] --
require 'torch'
require 'string'

-- construct an object to deal with the mapping
local lexicon = torch.class('Lexicon')

function lexicon:__init(lexPath)
    assert(paths.filep(lexPath), lexPath ..' not found')

    self.word2tokens = {}

    -- make maps
    for line in io.lines(lexPath) do
        line = string.lower(line)
        entry = string.split(line, "%s+")
        word = entry[1]
        if word:match("2") then
            word = word:match("(%a+)")
        end
        head = table.remove(entry, 1)
        phones = table.concat(entry," ")
        -- print(word,phones)
        self.word2tokens[word] = phones
    end
end
