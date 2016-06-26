util = require '../Utils'

assert(#arg == 1)

local out_dir = './' -- path to save lmdb
-- Window size and stride for the spectrogram transformation.
-- Use a 20ms window and 10ms stride. Since audio is 16khz, we multiply by 16,000
local windowSize = 0.02 * 16000
local stride = 0.01 * 16000

util.mk_lmdb(arg[1], 'train_index.txt', '../dictionary',
			out_dir..'train', windowSize, stride)

util.mk_lmdb(arg[1], 'test_index.txt', '../dictionary',
			out_dir..'test', windowSize, stride)