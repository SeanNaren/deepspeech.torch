require 'torch'
local cmd = torch.CmdLine()
cmd:option('-rootPath', 'an4', 'Path to the an4 root')
cmd:option('-newPath', 'an4_dataset', 'Path to the new data path')
cmd:option('-audioExtension', 'sph', 'The extension of the audio files (wav/mp3/sph/etc)')
cmd:option('-move', false, 'Moves the files over rather than copies, used to save space')

local opt = cmd:parse(arg)

local an4TestPath = opt.rootPath .. '/etc/an4_test.'
local an4TrainPath = opt.rootPath .. '/etc/an4_train.'
local an4AudioPath = opt.rootPath .. '/wav'

-- strips down the transcripts into pure text
local function processText(line)
    local text = line:gsub('<s>', ''):gsub('</s>', ''):gsub('^%s', ''):gsub('%(.*%)', ''):gsub('%s*$', '')
    return text
end

local function createDataset(pathToAN4, an4AudioPath, newPath)
    sys.execute("mkdir " .. newPath)
    local fileids = pathToAN4 .. 'fileids'
    local transcripts = pathToAN4 .. 'transcription'
    local filePaths = {}
    for filePath in io.lines(fileids) do
        table.insert(filePaths, filePath)
    end
    local counter = 1
    for line in io.lines(transcripts) do
        local text = processText(line)
        local filePath = filePaths[counter]
        -- new filename extracted from an4 file id
        local fileName = sys.split(filePath, '/')[3] -- last part is the filename
        -- create new text file with clean transcript
        local textPath = newPath .. '/' .. fileName .. '.txt'
        local file = io.open(textPath, "w")
        file:write(text)
        file:close()
        -- move audio to correct place
        local audioPath = an4AudioPath .. '/' .. filePath .. '.' .. opt.audioExtension
        local newPath = newPath .. '/' .. fileName .. '.' .. opt.audioExtension
        local command
        if opt.move then command = "mv " else command = "cp " end
        sys.execute(command .. audioPath .. ' ' .. newPath)
        counter = counter + 1
    end
end

sys.execute("mkdir " .. opt.newPath)
createDataset(an4TrainPath, an4AudioPath, opt.newPath .. '/train/')
createDataset(an4TestPath, an4AudioPath, opt.newPath .. '/test/')
