#!/bin/sh
chmod u+rx ./ConvertAN4ToWav.sh ./generateIndices.sh
wget http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz
tar -xzvf an4_raw.bigendian.tar.gz
rm -r an4_raw.bigendian.tar.gz
ln -s ../Mapper.lua .
AN4_PATH='an4'
echo "ROOT_FOLDER: $AN4_PATH"
find $AN4_PATH -name '*.wav' -delete
echo "Converting raw an4 dataset..."
./ConvertAN4ToWav.sh $AN4_PATH
echo "Generating Indices..."
./generateIndices.sh $AN4_PATH
echo "Generating LMDB..."
th generateLMDB.lua $AN4_PATH/wav/