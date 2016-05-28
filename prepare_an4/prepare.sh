#!/bin/sh
chmod u+rx ./ConvertAN4ToWav.sh ./gen_index.sh
wget http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz
tar -xzvf an4_raw.bigendian.tar.gz
rm -r an4_raw.bigendian.tar.gz
ln -s ../mapper.lua .
AN4_PATH='an4'
echo "ROOT_FOLDER: $AN4_PATH"
find $AN4_PATH -name '*.wav' -delete
echo "Convert..."
./ConvertAN4ToWav.sh $AN4_PATH
echo "Generate Index..."
./gen_index.sh $AN4_PATH
echo "Generate LMDB..."
th gen_lmdb.lua $AN4_PATH/wav/