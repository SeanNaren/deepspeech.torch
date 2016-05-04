-- Read a transcripts file and convert to phones

local AN4PhonemeDictionary = {}

local lookuptable = {}

local function fileexists(file)
    if (file == nil) then return false end
    local f = io.open(file,'r')
    if (f ~= nil) then io.close(f); return true else return false end
end

local function loaddictionary(dictfile)
    local word, phones
    lookuptable = {}
    for entry in io.lines(dictfile) do
        entry = string.lower(entry)
        _, _, word, phones = string.find(entry,"(%S+) (.+)")
        if (word ~= nil and phones ~= nil) then
            lookuptable[word] = phones
        end
    end
end

function AN4PhonemeDictionary.init(dictfile)
    assert(fileexists(dictfile), "Cannot open supplied dictionary")
    loaddictionary(dictfile)
end

function AN4PhonemeDictionary.LookUpWord(word)
    if (word == nil) then return nil end
    word = string.lower(word)
    return lookuptable[word]
end

function AN4PhonemeDictionary.PrintSample()
    local i = 1
    for word,phones in pairs(lookuptable) do
        io.write(string.format("%s: %s\n", word, phones))
        i = i+1
        if (i > 10) then break end
    end

end

return AN4PhonemeDictionary