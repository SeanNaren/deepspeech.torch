--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
require 'xlua'
require 'lmdb'
require 'torch'


-- manipulate with this object
local util = {}


-- construct an object to deal with the mapping
local mapper = torch.class('mapper')

function mapper:__init(dict_path)
    self.alphabet2token = {}
    self.token2alphabet = {}
    
    -- make maps
    local cnt = 0
    for line in io.lines(dict_path) do
        self.alphabet2token[line] = cnt
        self.token2alphabet[cnt] = line
        cnt = cnt + 1
    end
end


local function split(s, p)
    local rt= {}
    string.gsub(s, '[^'..p..']+', function(w) table.insert(rt, w) end )
    return rt
end


local function trans2tokens(line, _mapper)
    --[[
        input:
            line: ERASE C Q Q F SEVEN (id)

        output:
            label: {3,7,1,2,8}
            line: erase c q q f seven
    --]]

    local label = {}
    line = string.lower(line)
    -- Remove: beginning space, BOS, EOS, fileid, final space, string ids (<s> and </s>).
    line = line:gsub('^%s', ''):gsub('', ''):gsub('', ''):gsub('%(.+%)', ''):gsub('%s$', ''):gsub('<s>', ''):gsub('</s>', '')
    --Remove the space at the end of the line.
    for i = 1, #line do
        local character = line:sub(i, i)
        table.insert(label, _mapper.alphabet2token[character])
    end

    return torch.serialize(label), torch.serialize(line)
end


local function start_txn(_path, _name)
    local db = lmdb.env{
        Path = _path,
        Name = _name
    }
    db:open()
    local txn = db:txn()
    return db, txn
end

local function end_txn(db, txn)
    txn:commit()
    txn:abort()
    db:close()
end 

-- =============================
-- Main entrance
-- =============================
function util.mk_lmdb(index_path, dict_path, out_dir, windowSize, stride)
    --[[
        read index and dict files and make lmdb

        input:
            index_path: path to the index file
            dict_path: path to the dict file
            out_dir: dir to store lmdb
            windowSize, stride: hyperparas for making spects

        NOTE:
            line sturct of dict file: <char/word>

            line struct of index file: <wave_file_path>@<transcript>@,
            where wave_file_path should be absolute path
    --]]

    local mapper = mapper(dict_path)
    
    -- start writing
    local db_spect, txn_spect = start_txn(out_dir..'/train_spect', 'train_spect')
    local db_label, txn_label = start_txn(out_dir..'/train_label', 'train_label')
    local db_trans, txn_trans = start_txn(out_dir..'/train_trans', 'train_trans')
    
    local cnt = 1
    for line in io.lines(index_path) do
        print('processing ' .. line .. ' cnt: ' .. cnt)
        local wave_path, trans = split(line,'@')[1], split(line,'@')[2]

        -- make label
        local label, modified_trans = trans2tokens(trans, mapper)

        -- make spect
        local wave = audio.load(wave_path)
        local spect = audio.spectrogram(wave, windowSize, 'hamming', stride) -- freq-by-frames tensor
        
        -- put into lmdb
        txn_spect:put(cnt, spect:byte())
        txn_label:put(cnt, label)
        txn_trans:put(cnt, modified_trans)

        cnt = cnt + 1
    end
    
    -- close
    end_txn(db_spect, txn_spect)
    end_txn(db_label, txn_label)
    end_txn(db_trans, txn_trans)

end

return util
