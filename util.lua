--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
require 'xlua'
require 'lmdb'
-- manipulate with this object
local util = {}


local alphabet = {
    '$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '
}

local alphabetMapping = {}
local indexMapping = {}
for index, character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

--Given an index returns the letter at that index.
function util.findLetter(index)
    return indexMapping[index]
end


local function split(s, p)
    local rt= {}
    string.gsub(s, '[^'..p..']+', function(w) table.insert(rt, w) end )
    return rt
end


local function trans2token(labels, transcripts, line)
    --[[
        input:
            line: ERASE C Q Q F SEVEN (id)
        output:
            label: {3,7,1,2,8}
            transcript: erase c q q f seven
    --]]

    local label = {}
    line = string.lower(line)
    -- Remove: beginning space, BOS, EOS, fileid, final space, string ids (<s> and </s>).
    line = line:gsub('^%s', ''):gsub('', ''):gsub('', ''):gsub('%(.+%)', ''):gsub('%s$', ''):gsub('<s>', ''):gsub('</s>', '')
    --Remove the space at the end of the line.
    for i = 1, #line do
        local character = line:sub(i, i)
        table.insert(label, alphabetMapping[character])
    end
    table.insert(labels, torch.serialize(label)) -- serialize to store in lmdb
    table.insert(transcripts, torch.serialize(line))
end


local function table2lmdb(_path, _name, _table)
    print('writing '.._name..' len: '..#_table)
    print(_table[1])
    local db = lmdb.env{
        Path = _path,
        Name = _name
    }
    db:open()
    local txn = db:txn()

    for i=1,#_table do
        txn:put(i, _table[i]) -- obejcts are all previously serialized
    end
    txn:commit()
    txn:abort()
    db:close()
end

-- =============================
-- Main entrance
-- =============================
function util.an4train_2lmdb(_dir, out_dir, windowSize, stride)
    --[[
        input:
            _dir: an4 dir
            out_dir: out dir of lmdb
    --]]
    local wave_paths = _dir .. "/etc/an4_train.fileids"
    local trans_path = _dir .. "/etc/an4_train.transcription"
    
    local transcripts = {} -- {'abc bac', 'abc', ..} each is trans
    local labels = {} -- {{1,2,3}, {1,2,3}, ..} each is a label for an utterance
    local spects = {} -- {t1, t2, ..}
    
    -- text to token
    for line in io.lines(trans_path) do
        trans2token(labels, transcripts, line)
    end
    
    -- wave to specturm
    for filename in io.lines(wave_paths) do
        print('doing '..filename)
        local wave = audio.load(_dir .. "/wav/" .. filename .. ".wav")
        local spect = audio.spectrogram(wave, windowSize, 'hamming', stride) -- freq-by-frames tensor
        table.insert(spects, spect:byte())
    end 

    --store in lmdb
    table2lmdb(out_dir..'/train_spect', 'train_spec', spects)
    table2lmdb(out_dir..'/train_label', 'train_label', labels)
    table2lmdb(out_dir..'/train_trans', 'train_trans', transcripts)
end



function util.mk_lmdb(file_path, out_dir, windowSize, stride)
    --[[
        read an index file and make lmdb

        input:
            file_path: path to the index file
            out_dir: dir to store lmdb
            windowSize, stride: hyperparas for making spects

        NOTE:
            struct of index file: <wave_file_path>@<transcript>@,
            where wave_file_path should be absolute path
    --]]
    
    local transcripts = {} -- {'abc bac', 'abc', ..} each is trans
    local labels = {} -- {{1,2,3}, {1,2,3}, ..} each is a label for an utterance
    local spects = {} -- {t1, t2, ..}

    
    for line in io.lines(file_path) do
        local wave_path, trans = split(line,'@')[1], split(line,'@')[2]

        -- make label
        trans2token(labels, transcripts, trans)
        
        -- make spect
        local wave = audio.load(wave_path)
        local spect = audio.spectrogram(wave, windowSize, 'hamming', stride) -- freq-by-frames tensor
        table.insert(spects, spect:byte())
    end
     
    --store in lmdb
    table2lmdb(out_dir..'/train_spect', 'train_spec', spects)
    table2lmdb(out_dir..'/train_label', 'train_label', labels)
    table2lmdb(out_dir..'/train_trans', 'train_trans', transcripts)

end


function util.retrieveAN4TestDataSet(_dir, windowSize, stride, numberOfSamples)
    local wave_paths = _dir .. "/etc/an4_test.fileids"
    local trans_path = _dir .. "/etc/an4_test.transcription"
    local nbSamples = numberOfSamples or 130 -- Amount of samples (defaults to all).
    local labels = {}
    local transcripts = {} -- For verification of words, we return the lines of the test data.

    for line in io.lines(trans_path) do
        trans2token(labels, transcripts, line)
    end
    local dataSet = an4Dataset(_dir, wave_paths, labels, transcripts, windowSize, stride, nbSamples)
    return dataSet, transcripts
end



return util
