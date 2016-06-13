require 'nn'
require 'torch'
require 'lmdb'
require 'xlua'
require 'paths'
tds = require 'tds'
local model_t = require 'DeepSpeechModel'
local cal_size = model_t[2]
local get_min_width = model_t[3]

--[[

    this file defines Loader and loader:
        - Loader returns different inds of nxt btach
        - loader loads data from lmdb given the inds

    NOTE:
        - make sure calculateInputSizes() in DeepSpeechModel.lua is set correctly

--]]

torch.setdefaulttensortype('torch.FloatTensor')

local Loader = torch.class('Loader')

function Loader:__init(_dir, batch_size)

    -- constants to indicate the loading style
    self.DEFAULT = 1
    self.SAMELEN = 2
    self.SORTED = 3

    self.db_spect = lmdb.env { Path = _dir .. '/spect', Name = 'spect' }
    self.db_label = lmdb.env { Path = _dir .. '/label', Name = 'label' }
    self.db_trans = lmdb.env { Path = _dir .. '/trans', Name = 'trans' }
    self._dir = _dir

    self.batch_size = batch_size
    self.cnt = 1

    -- get the size of lmdb
    self.db_spect:open()
    self.db_label:open()
    self.db_trans:open()
    local l1 = self.db_spect:stat()['entries']
    local l2 = self.db_label:stat()['entries']
    local l3 = self.db_trans:stat()['entries']

    assert(l1 == l2 and l2 == l3, 'data sizes in each lmdb must agree')

    self.lmdb_size = l1

    self.db_spect:close()
    self.db_label:close()
    self.db_trans:close()

    assert(self.lmdb_size > self.batch_size, 'batch_size larger than lmdb_size')
    self.sorted_inds = {}
    self.len_num = 0 -- number of unique seqLengths
    self.min_width = get_min_width() --from DeepSpeech
end

function Loader:prep_sorted_inds()
    --[[
        prep a table for sorted inds, can detect previously saved table in lmdb folder
    --]]

    print('preparing sorted indices..')
    local indicesFilePath = self._dir .. '/' .. 'sorted_inds_' .. self.min_width

    -- check if there is previously saved inds
    if paths.filep(indicesFilePath) then
        print('found previously saved inds..')
        self.sorted_inds = torch.load(indicesFilePath)
        print('original size: '..self.lmdb_size..' valid data: '..#self.sorted_inds)
        self.lmdb_size = #self.sorted_inds
        return
    end

    -- if not make a new one
    print('did not find previously saved indices, generating.')

    self.db_spect:open(); local txn = self.db_spect:txn(true)
    self.db_label:open(); local txn_label = self.db_label:txn(true)

    local lengths = {}
    -- those shorter than min_width are ignored
    local true_size = 0
    for i = 1, self.lmdb_size do
        local lengthOfAudio = txn:get(i):size(2) -- get the len of spect
        local lengthOfLabel = #(torch.deserialize(txn_label:get(i)))

        if lengthOfAudio >= self.min_width and cal_size(lengthOfAudio) >= lengthOfLabel then
            true_size = true_size + 1
            table.insert(self.sorted_inds, { i, lengthOfAudio })
            if lengths[lengthOfAudio] == nil then lengths[lengthOfAudio] = true end
            if i % 100 == 0 then xlua.progress(i, self.lmdb_size) end
        end
    end
    

    print('original size: '..self.lmdb_size..' valid data: '..true_size)
    self.lmdb_size = true_size -- set size to true size
    txn:abort(); self.db_spect:close()
    txn_label:abort(); self.db_label:close()

    local function comp(a, b) return a[2] < b[2] end
    table.sort(self.sorted_inds, comp)

    for _ in pairs(lengths) do self.len_num = self.len_num + 1 end -- number of different seqLengths
    torch.save(indicesFilePath, self.sorted_inds)
end


function Loader:nxt_sorted_inds()
    local meta_inds = self:nxt_inds()
    local inds = {}
    for _, v in ipairs(meta_inds) do
        table.insert(inds, self.sorted_inds[v][1])
    end
    return inds
end


function Loader:nxt_same_len_inds()
    --[[
        return inds with same seqLength, a solution before zero-masking can work
    --]]

    local _len = self.sorted_inds[self.cnt][2]
    local inds = {}
    while (self.cnt <= self.lmdb_size and self.sorted_inds[self.cnt][2] == _len) do
        -- NOTE: true index store in table, instead of cnt
        table.insert(inds, self.sorted_inds[self.cnt][1])
        self.cnt = self.cnt + 1
    end

    if self.cnt > self.lmdb_size then self.cnt = 1 end

    return inds
end

function Loader:nxt_inds()
    --[[
        return indices of the next batch
    --]]

    local inds = {}
    if self.lmdb_size > self.cnt + self.batch_size - 1 then
        for i = 0, self.batch_size - 1 do
            table.insert(inds, self.cnt + i)
        end

        self.cnt = self.cnt + self.batch_size
        return inds
    end

    -- case where cnt+size >= total size
    for i = self.cnt, self.lmdb_size do
        table.insert(inds, i)
    end

    self.cnt = self.batch_size - (self.lmdb_size - self.cnt)
    for i = 1, self.cnt - 1 do -- overflow inds
    table.insert(inds, i)
    end

    return inds
end

function Loader:nxt_batch(mode, flag)
    --[[
        return a batch by loading from lmdb just-in-time

        input:
            mode: should be Loader.DEFAULT/SAMELEN/SORTED; USE ONLY ONE MODE FOR ONE TRAINING
            flag: indicates whether to load trans

        TODO we allocate 2 * batch_size space
    --]]

    ---------------------------- get inds -------------------------------------
    local inds
    if mode == self.DEFAULT then
        return self:nxt_default_batch(flag)
    else
        assert(#self.sorted_inds > 0, 'call prep_sorted_inds before nxt_batch')
        if mode == self.SAMELEN then
            inds = self:nxt_same_len_inds()
        elseif mode == self.SORTED then
            inds = self:nxt_sorted_inds()
        end
    end

    --------------------------- loading --------------------------------------
    local tensor_list = tds.Vec()
    local label_list = {}
    local max_w = 0
    local h = 0

    local trans_list = {}
    local txn_trans

    self.db_spect:open(); local txn_spect = self.db_spect:txn(true) -- readonly
    self.db_label:open(); local txn_label = self.db_label:txn(true)
    if flag then self.db_trans:open(); txn_trans = self.db_trans:txn(true) end

    local sizes_array = torch.Tensor(#inds)
    local cnt = 1
    -- reads out a batch and store in lists
    for _, ind in next, inds, nil do
        local tensor = txn_spect:get(ind)
        local label = torch.deserialize(txn_label:get(ind))

        h = tensor:size(1)
        sizes_array[cnt] = tensor:size(2); cnt = cnt + 1 -- record true length
        if max_w < tensor:size(2) then max_w = tensor:size(2) end -- find the max len in this batch

        tensor_list:insert(tensor)
        table.insert(label_list, label)
        if flag then table.insert(trans_list, torch.deserialize(txn_trans:get(ind))) end
    end

    -- store tensors into a fixed len tensor_array TODO should find a better way to do this
    local tensor_array = torch.Tensor(#inds, 1, h, max_w):zero()
    for ind, tensor in ipairs(tensor_list) do
        tensor_array[ind][1]:narrow(2, 1, tensor:size(2)):copy(tensor)
    end

    txn_spect:abort(); self.db_spect:close()
    txn_label:abort(); self.db_label:close()
    if flag then txn_trans:abort(); self.db_trans:close() end

    if flag then return tensor_array, label_list, sizes_array, trans_list end
    return tensor_array, label_list, sizes_array
end

function Loader:nxt_default_batch(flag)

    local tensor_list = tds.Vec()
    local label_list = {}
    local max_w = 0
    local h = 0
    local trans_list = {}
    local sizes_list = {}
    local txn_trans

    self.db_spect:open(); local txn_spect = self.db_spect:txn(true) -- readonly
    self.db_label:open(); local txn_label = self.db_label:txn(true)
    if flag then self.db_trans:open(); txn_trans = self.db_trans:txn(true) end

    -- reads out a batch and store in lists
    local batch_cnt = 0
    while batch_cnt < self.batch_size do
        
        local tensor = txn_spect:get(self.cnt)
        local label = torch.deserialize(txn_label:get(self.cnt))
        local width = tensor:size(2)

        if width >= self.min_width and cal_size(width) >= #label then
            batch_cnt = batch_cnt + 1

            h = tensor:size(1)
            table.insert(sizes_list, width)
            if max_w < width then max_w = width end -- find the max len in this batch

            tensor_list:insert(tensor)
            table.insert(label_list, label)
            if flag then table.insert(trans_list, torch.deserialize(txn_trans:get(self.cnt))) end
        end

        self.cnt = self.cnt + 1
        if self.cnt > self.lmdb_size then self.cnt = self.cnt % self.lmdb_size end
    end

    -- store tensors into a fixed len tensor_array TODO should find a better way to do this
    local tensor_array = torch.Tensor(batch_cnt, 1, h, max_w):zero()
    for ind, tensor in ipairs(tensor_list) do
        tensor_array[ind][1]:narrow(2, 1, tensor:size(2)):copy(tensor) --TODO wrong
    end

    txn_spect:abort(); self.db_spect:close()
    txn_label:abort(); self.db_label:close()
    if flag then txn_trans:abort(); self.db_trans:close() end

    local sizes_array = torch.Tensor(#sizes_list)
    for k,v in ipairs(sizes_list) do sizes_array[k] = v end

    if flag then return tensor_array, label_list, sizes_array, trans_list end
    return tensor_array, label_list, sizes_array
end
