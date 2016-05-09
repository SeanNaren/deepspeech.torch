#!/bin/sh
if [ $# -gt 1 ] ; then
	echo "USAGE: $0 [ROOT_PATH]"
	exit 1;
fi

if [ $# = 1 ] ; then
	AN4_PATH=$1
else
	AN4_PATH=''
fi

echo "ROOT_FOLDER: $AN4_PATH"
echo "Convert..."
./ConvertAN4ToWav.sh $AN4_PATH
echo "Generate Index..."
./gen_index.sh $AN4_PATH
echo "Generate LMDB..."
th gen_lmdb.lua $AN4_PATH/wav/