require 'rnn'
require 'nngraph'
function makeDataParallel(model, nGPU)
    if nGPU > 0 then
        cudnn.fastest = true
        local function BatchNorm(module)
            return torch.type(module):find('BatchNormalization')
        end
        model = cudnn.convert(model, cudnn, BatchNorm)
        if nGPU > 1 then
            gpus = torch.range(1, nGPU):totable()
            dpt = nn.DataParallelTable(1):add(model, gpus):threads(function()
                require 'nngraph'
                require 'cudnn'
                cudnn.fastest = true
                require 'BatchBRNNReLU'
            end)
            dpt.gradInput = nil
            model = dpt
        end
        model:cuda()
    end
    return model
end

local function cleanDPT(module, device)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(device or 1)
    newDPT:add(module:get(1), device or 1)
    return newDPT
end

function saveDataParallel(modelPath, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(modelPath, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module))
            else
                temp_model:add(module)
            end
        end
        torch.save(modelPath, temp_model)
    elseif torch.type(model) == 'nn.gModule' then
        torch.save(modelPath, model)
    else
        error('This saving function only works with Sequential or DataParallelTable modules.')
    end
end

function loadDataParallel(modelPath, nGPU)
    if nGPU > 1 then
        require 'cudnn'
        require 'BatchBRNNReLU'
    end
    local model = torch.load(modelPath)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1):float(), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
            end
        end
        return model
    elseif torch.type(model) == 'nn.gModule' then
        model = makeDataParallel(model, nGPU)
        return model
    else
        error('The loaded model is not a Sequential or DataParallelTable module.')
    end
end