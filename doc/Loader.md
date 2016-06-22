# Loader

Defines the indexer class and the loader class, handling batching of the dataset to train the network.

## Indexer

Handles returning the next indices of the batch to load into memory, to train the network with.

### indexer:__init(_dir, batchSize)

`dirPath` Directory containing the LMDB data folders for spectrogram, labels and transcripts.

`batchSize` The sizes of each batch to create.

### indexer:prep_sorted_inds()

Sorts the pointers of each training sample by length, and stores the order in the class for retrieving batches.

### indexer:nxt_sorted_inds()

Returns the next set of sorted indices that can be loaded from disk based on the batch size  (iterator).

### indexer:nxt_same_len_inds()

Returns the next set of indices that can be loaded from disk that have the same length (iterator).

### indexer:nxt_inds()

Returns the next set of indices based purely on batch size (no ordering).

## Loader

Loads batches of data from LMDB files used in training/testing.

### Loader:__init(dirPath)

`dirPath` Directory containing the LMDB data folders for spectrogram, labels and transcripts.

### Loader:nxt_batch(indices, includeTranscripts)

Returns the next batch of the dataset based on the given indices.

`indices` The indices of the test samples that need to be retrieved. This is handled by the Indexer class above.

`includeTranscripts` Set to true if transcripts are needed as well for each training sample.