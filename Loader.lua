require 'nn'
require 'torch'
require 'lmdb'
require 'xlua'
require 'paths'
require 'Mapper'
local tds = require 'tds'

torch.setdefaulttensortype('torch.FloatTensor')

local indexer = torch.class('indexer')

function indexer:__init(dirPath, batchSize)

    local dbSpect = lmdb.env { Path = dirPath .. '/spect', Name = 'spect' }
    local dbTrans = lmdb.env { Path = dirPath .. '/trans', Name = 'trans' }

    self.batchSize = batchSize
    self.count = 1
    -- get the size of lmdb
    dbSpect:open()
    dbTrans:open()
    local audioLMDBSize = dbSpect:stat()['entries']
    local transcriptLMDBSize = dbTrans:stat()['entries']
    self.size = audioLMDBSize
    dbSpect:close()
    dbTrans:close()
    self.nbOfBatches = math.ceil(self.size / self.batchSize)
    assert(audioLMDBSize == transcriptLMDBSize, 'Audio and transcript LMDBs had different lengths!')
    assert(self.size > self.batchSize, 'batchSize larger than lmdb size!')

    self.inds = torch.range(1, self.size):split(batchSize)
    self.batchIndices = torch.range(1, self.nbOfBatches)
end

function indexer:nextIndices()
    if self.count > #self.inds then self.count = 1 end
    local index = self.batchIndices[self.count]
    local inds = self.inds[index]
    self.count = self.count + 1
    return inds
end

function indexer:permuteBatchOrder()
    self.batchIndices = torch.randperm(self.nbOfBatches)
end

local Loader = torch.class('Loader')

function Loader:__init(dirPath, mapper)
    self.dbSpect = lmdb.env { Path = dirPath .. '/spect', Name = 'spect' }
    self.dbTrans = lmdb.env { Path = dirPath .. '/trans', Name = 'trans' }
    self.dbSpect:open()
    self.size = self.dbSpect:stat()['entries']
    self.dbSpect:close()
    self.mapper = mapper
end

function Loader:nextBatch(indices)
    local tensors = tds.Vec()
    local targets = {}
    local transcripts = {}

    local maxLength = 0
    local freq = 0

    self.dbSpect:open(); local readerSpect = self.dbSpect:txn(true) -- readonly
    self.dbTrans:open(); local readerTrans = self.dbTrans:txn(true)

    local size = indices:size(1)

    local sizes = torch.Tensor(#indices)

    local permutedIndices = torch.randperm(size) -- batch tensor has different order each time
    -- reads out a batch and store in lists
    for x = 1, size do
        local ind = indices[permutedIndices[x]]
        local tensor = readerSpect:get(ind):float()
        local transcript = readerTrans:get(ind)

        freq = tensor:size(1)
        sizes[x] = tensor:size(2)
        if maxLength < tensor:size(2) then maxLength = tensor:size(2) end -- find the max len in this batch

        tensors:insert(tensor)
        table.insert(targets, self.mapper:encodeString(transcript))
        table.insert(transcripts, transcript)
    end

    local inputs = torch.Tensor(size, 1, freq, maxLength):zero()
    for ind, tensor in ipairs(tensors) do
        inputs[ind][1]:narrow(2, 1, tensor:size(2)):copy(tensor)
    end

    readerSpect:abort(); self.dbSpect:close()
    readerTrans:abort(); self.dbTrans:close()

    return inputs, targets, sizes, transcripts
end
