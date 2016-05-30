util = require '../Util'

assert(#arg == 1)

local out_dir = './' -- path to save lmdb
--Window size and stride for the spectrogram transformation.
local windowSize = 256
local stride = 75

util.mk_lmdb(arg[1], 'train_index.txt', '../dictionary',
			out_dir..'train', windowSize, stride)

util.mk_lmdb(arg[1], 'test_index.txt', '../dictionary',
			out_dir..'test', windowSize, stride)