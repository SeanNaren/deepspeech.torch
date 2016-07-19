# Utils

Contains methods to access LMDB database and convert the AN4 dataset into the appropriate format.

### util.mk_lmdb(root_path, index_path, dict_path, out_dir, windowSize, stride)

Used in the `prepare.sh` script to create the AN4 training and test dataset for our neural net. Converts the
given directory into LMDB databases that contain the data used in online training/testing.

`index_path` Path to the index file.

`dict_path` Path to the dictionary file.

`out_dir` Directory where the LMDBs are stored.

`windowSize`, `stride` Parameters chosen for the spectrogram.

## Index file

The index file contains a list of file paths associated to a transcript like below:

```lua
<wave_file_path>@<transcript>@
an4/test/example.wav@EXAMPLE TRANSCRIPT@
```

The ```@``` symbols are important to add for each entry. More information about extending to your own dataset can be
seen [here](https://github.com/SeanNaren/CTCSpeechRecognition/wiki/Adding-Custom-Datasets).