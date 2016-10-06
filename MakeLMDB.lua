-- Expects data in the format of <root><train/test><datasetname><filename.wav/filename.txt>
-- Creates an LMDB of everything in these folders into a train and test set.

require 'lfs'
require 'audio'
require 'xlua'
require 'lmdb'
require 'torch'
require 'parallel'

local tds = require 'tds'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'prepare_datasets/an4_dataset', 'Path to the data')
cmd:option('-lmdbPath', 'prepare_datasets/an4_lmdb', 'Path to save LMDBs to')
cmd:option('-windowSize', 0.02, 'Window size for audio data')
cmd:option('-stride', 0.01, 'Stride for audio data')
cmd:option('-sampleRate', 16000, 'Sample rate of audio data (Default 16khz)')
cmd:option('-audioExtension', 'sph', 'The extension of the audio files (wav/mp3/sph/etc)')
cmd:option('-processes', 8, 'Number of processes used to create LMDB')

local opt = cmd:parse(arg)
local dataPath = opt.rootPath
local lmdbPath = opt.lmdbPath
local extension = '.' .. opt.audioExtension
parallel.nfork(opt.processes)

local function startWriter(path, name)
    local db = lmdb.env {
        Path = path,
        Name = name
    }
    db:open()
    local txn = db:txn()
    return db, txn
end

local function closeWriter(db, txn)
    txn:commit()
    db:close()
end

local function createLMDB(dataPath, lmdbPath)
    local vecs = tds.Vec()

    local function file_exists(name)
        local f = io.open(name, "r")
        if f ~= nil then io.close(f) return true else return false end
    end

    if not file_exists('sort_ids.t7') then
        local size = tonumber(sys.execute("find " .. dataPath .. " -type f -name '*'" .. extension .. " | wc -l "))
        vecs:resize(size)

        local files = io.popen("find -L " .. dataPath .. " -type f -name '*" .. extension .. "'")
        local counter = 1
        print("Retrieving sizes for sorting...")
        local buffer = tds.Vec()
        buffer:resize(size)

        for file in files:lines() do
            buffer[counter] = file
            counter = counter + 1
        end

        local function getSize(opts)
            local audioFilePath = opts.file
            local transcriptFilePath = opts.file:gsub(opts.extension, ".txt")
            local opt = opts.opt
            local audioFile = audio.load(audioFilePath)
            local length = audio.spectrogram(audioFile, opt.windowSize * opt.sampleRate, 'hamming', opt.stride * opt.sampleRate):size(2)
            return { audioFilePath, transcriptFilePath, length }
        end

        for x = 1, opt.processes do
            local opts = { extension = extension, file = buffer[x], opt = opt }
            parallel.children[x]:send({ opts, getSize })
        end

        local processCounter = 1
        for x = 1, size do
            local result = parallel.children[processCounter]:receive()
            vecs[x] = tds.Vec(unpack(result))
            xlua.progress(x, size)
            if x % 1000 == 0 then collectgarbage() end
            -- send next index to retrieve
            if x + opt.processes <= size then
                local opts = { extension = extension, file = buffer[x + opt.processes], opt = opt }
                parallel.children[processCounter]:send({ opts, getSize })
            end
            if processCounter == opt.processes then
                processCounter = 1
            else
                processCounter = processCounter + 1
            end
        end
        print("Sorting...")
        -- sort the files by length
        local function comp(a, b) return a[3] < b[3] end

        vecs:sort(comp)
        torch.save('sort_ids.t7', vecs)
    else
        vecs = torch.load('sort_ids.t7')
    end
    local size = #vecs

    print("Creating LMDB dataset to: " .. lmdbPath)
    -- start writing
    local dbSpect, readerSpect = startWriter(lmdbPath .. '/spect', 'spect')
    local dbTrans, readerTrans = startWriter(lmdbPath .. '/trans', 'trans')

    local function getData(opts)
        local opt = opts.opt
        local audioFile = audio.load(opts.audioFilePath)
        local spect = audio.spectrogram(audioFile, opt.windowSize * opt.sampleRate, 'hamming', opt.stride * opt.sampleRate) -- freq-by-frames tensor

        -- put into lmdb
        spect = spect:float()

        -- normalize the data
        local mean = spect:mean()
        local std = spect:std()
        spect:add(-mean)
        spect:div(std)

        local transcript
        for line in io.lines(opts.transcriptFilePath) do
            transcript = line
        end
        return { spect, transcript }
    end

    for x = 1, opt.processes do
        local vec = vecs[x]
        local opts = { audioFilePath = vec[1], transcriptFilePath = vec[2], opt = opt }
        parallel.children[x]:send({ opts, getData })
    end

    local processCounter = 1
    for x = 1, size do
        local result = parallel.children[processCounter]:receive()
        local spect, transcript = unpack(result)

        readerSpect:put(x, spect)
        readerTrans:put(x, transcript)

        -- commit buffer
        if x % 1000 == 0 then
            readerSpect:commit(); readerSpect = dbSpect:txn()
            readerTrans:commit(); readerTrans = dbTrans:txn()
        end
        if x % 5000 == 0 then
            collectgarbage()
        end

        if x + opt.processes <= size then
            local vec = vecs[x + opt.processes]
            local opts = { audioFilePath = vec[1], transcriptFilePath = vec[2], opt = opt }
            parallel.children[processCounter]:send({ opts, getData })
        end
        if processCounter == opt.processes then
            processCounter = 1
        else
            processCounter = processCounter + 1
        end
        xlua.progress(x, size)
    end

    closeWriter(dbSpect, readerSpect)
    closeWriter(dbTrans, readerTrans)
end

function parent()
    local function looper()
        require 'torch'
        require 'audio'
        while true do
            local object = parallel.parent:receive()
            local opts, code = unpack(object)
            local result = code(opts)
            collectgarbage()
            parallel.parent:send(result)
        end
    end

    parallel.children:exec(looper)

    createLMDB(dataPath .. '/train', lmdbPath .. '/train')
    createLMDB(dataPath .. '/test', lmdbPath .. '/test')
    parallel.close()
end

local ok, err = pcall(parent)
if not ok then
    print(err)
    parallel.close()
end