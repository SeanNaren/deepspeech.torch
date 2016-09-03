# Loader

Defines the indexer class and the loader class, handling batching of the dataset to train the network.

## Indexer

Handles returning the next indices of the batch to load into memory, to train the network with.

### indexer:__init(_dir, batchSize)

`dirPath` Directory containing the LMDB data folders for spectrogram, labels and transcripts.

`batchSize` The sizes of each batch to create.

### indexer:nextIndices()

Retrieves the next indices that need to be loaded by the loader from the LMDB dataset.

### indexer:permuteBatchOrder()

Permutes the batch order randomly. This is for the net to not train in sequence order every time.

## Loader

Loads batches of data from LMDB files used in training/testing.

### Loader:__init(dirPath)

`dirPath` Directory containing the LMDB data folders for spectrogram, labels and transcripts.

### Loader:nextBatch(indices)

Returns the next batch of the dataset based on the given indices.

`indices` The indices of the test samples that need to be retrieved. This is handled by the Indexer class above.