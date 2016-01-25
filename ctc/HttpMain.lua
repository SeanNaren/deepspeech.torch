local networkHandler = require('NetworkHandler')
local turbo = require("turbo")
local networks = {}
-- Create a new requesthandler with a method get() for HTTP GET.
local TrainNetworkRequestHandler = class("TrainNetworkRequestHandler", turbo.web.RequestHandler)
function TrainNetworkRequestHandler:post(networkName)
    local json = self:get_json(true)
    local net = networkHandler.trainNetwork(json)
    networks[networkName] = net
    self:write({ack = networkName .. " created"})
end
local PredictNetworkRequestHandler = class("PredictNetworkRequestHandler", turbo.web.RequestHandler)
function PredictNetworkRequestHandler:post(networkName)
    local json = self:get_json(true)
    local net = networks[networkName]
    local predictionResults = networkHandler.predictNetwork(net,json)
    print("predictions calculated")
    self:write({prediction = predictionResults})
end


-- Create an Application object and bind our HelloWorldHandler to the route '/hello'.
local app = turbo.web.Application:new({
    {"/api/train_network/(.*)$", TrainNetworkRequestHandler },
    {"/api/predict_network/(.*)$", PredictNetworkRequestHandler }
})
-- Set the server to listen on port 8888 and start the ioloop.
app:listen(5000)
turbo.ioloop.instance():start()

