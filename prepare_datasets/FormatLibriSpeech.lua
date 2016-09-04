require 'torch'
local threads = require 'threads'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'LibriSpeech', 'Path to the librispeech root')
cmd:option('-newPath', 'libri_dataset', 'Path to the new data path')
cmd:option('-audioExtension', 'flac', 'The extension of the audio files (wav/mp3/sph/etc)')
cmd:option('-move', false, 'Moves the files over rather than copies, used to save space')
cmd:option('-threads', 8, 'Number of threads to use')

local opt = cmd:parse(arg)
local extension = '.' .. opt.audioExtension

local libriTestPath = opt.rootPath .. '/test/'
local libriTrainPath = opt.rootPath .. '/train/'
local threads = threads.Threads(opt.threads, function(idx) require 'torch' require 'sys' end)

-- strips down the transcripts into pure text
local function processText(line)
    local text = line:gsub('[^a-zA-Z ]', '')
    return text
end

local function createDataset(libriPath, newDirPath)
    sys.execute("mkdir " .. newDirPath)
    local size = tonumber(sys.execute("find " .. libriPath .. " -type f -name '*'" .. extension .. " | wc -l "))

    local counter = 1

    local function formatData(line, dir)
        local text = processText(line)
        local id = line:match("([^ ]*) ") -- first part of transcript, used for audio file path and ID
        local audioFolders = sys.split(id, '-')

        local textPath = newDirPath .. '/' .. id .. '.txt'
        local file = io.open(textPath, "w")
        file:write(text)
        file:close()
        -- move audio to correct place
        local audioPath = dir .. '/' .. audioFolders[1] .. '/' .. audioFolders[2] .. '/' .. id .. extension
        local newPath = newDirPath .. '/' .. id .. extension
        local command
        if opt.move then command = "mv " else command = "cp " end
        sys.execute(command .. audioPath .. ' ' .. newPath)
    end

    local counter = 0

    local p = io.popen('find "' .. libriPath .. '" -maxdepth 1 -mindepth 1 -type d')
    for dir in p:lines() do
        local transcripts = io.popen("find -L " .. dir .. " -type f -name '*.txt'")
        for transcript in transcripts:lines() do
            for line in io.lines(transcript) do
                threads:addjob(function()
                    formatData(line, dir)
                end,
                    function()
                        counter = counter + 1
                        xlua.progress(counter, size)
                    end)
            end
        end
    end
end

sys.execute("mkdir " .. opt.newPath)
createDataset(libriTrainPath, opt.newPath .. '/train/')
createDataset(libriTestPath, opt.newPath .. '/test/')
