# Parameters for NEPAL
___
It is necessary to decide on a number of parameters before running the Neural Pattern Learning attack and the Graph Matching Attack.
The choice of parameter values can significantly impact attack performance, both in
terms of attack duration and success rate.

We have tried to come up with reasonable defaults that proved useful in our experiments. However,
your specific experiment might work better with other values.

The tables below describe the individual parameters, along with their default values
and relevant descriptions.
___

## NEPAL Configuration
**Argument Name:** `NEPAL_CONFIG`

| Parameter Name  | Description                                | Default |
|-----------------|--------------------------------------------|---------|
| ParallelTrials         | Number of Parallel Trials for Ray Tune Hyperparameter Optimization.          | 5   |
| TrainSize | Sets Training-Validation Split Ratio.            | 0.8       |
| Patience | Patience Epochs for Early Stopping.                    | 5      |
| MinDelta | Minimum improvement in the monitored metric required to reset early stopping patience. | 1e-4      |
| NumSamples   | Amount of Hyperparameter Optimization trials sampled from the search space.   | 125      |
| Epochs   | Maximum number of Epochs for Training.   | 25      |
| MetricToOptimize   | Metric to optimize for in the Hyperparameter Optimization (average_dice, average_precision, average_recall).   | "average_dice"      |
| MatchingTechnique   | Reconstruction Strategy (greedy only option, further extensions possible).   | "greedy"      |

___

## Global Configuration
**Argument Name:** `GLOBAL_CONFIG`

| Parameter Name | Description                                                                                                                                                                                     | Default                   |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| Data           | Dataset to run the attack on.                                                                                                                                                                   | "./data/titanic_full.tsv" |
| Overlap        | If GMA enabled: The share of overlapping records between the attacker's and the victim's data. Must be >0 and <=1. If GMA is disabled: the randomly sampled training proportion from the dataset (synthetic data split)                                                                                     | 1                         |
| DropFrom       | Which dataset should records be dropped from to achieve the desired overlap? One of "Eve" (Attacker), "Alice" (Victim) or "Both".                                                               | "Alice"                   |
| DevMode        | If True, additional dev data is saved                                              | False                     |
| BenchMode      | If True, the NEPAL attack is timed and duration of the attack is reported.                                          | True                     |
| Verbose        | If True, prints detailed status messages.                                                                                                                | True                      |
| MatchingMetric | Similarity metric to be computed on aligned embeddings during bipartite graph matching.                                                                  | "cosine"                  |
| Matching       | Matching algorithm for bipartite graph matching. Must be "MinWeight", "Stable", "Symmetric" or "NearestNeighbor".                                        | "MinWeight"               |                       |
| SaveAliceEncs | Stores a pickled dictionary containing UIDs as keys and encodings for Alice's Dataset.                            | False                     |
| SaveEveEncs   | Stores a pickled dictionary containing UIDs as keys and encodings for Eve's Dataset.                            | False                     |
| SaveResults   | Save NEPAL results for analysis script.   | True      |
| UseGPU   | Use GPU for CUDA.   | True      |
| SaveModel   | Save the final trained neural network model.   | False      |
| SavePredictions   | Save the predicted q-grams for each record with performance metrics.   | False      |
| GraphMatchingAttack   | Enable or Disable running the GMA to produce the known plaintext-encoding pairs. If disabled, a synthetic datasplit will be created   | False      |
___

## Encoding Configuration
**Argument Name:** `ENC_CONFIG`

| Parameter Name | Description                                                                                                        | Default                     |
|----------------|--------------------------------------------------------------------------------------------------------------------|-----------------------------|
| AliceAlgo      | Algorithm used for encoding Alice's data. One of "BloomFilter", "TabMinHash", "TwoStepHash" or None (No Encoding). | "TwoStepHash"               |
| AliceSecret    | Secret (seed for hash function selection/salt) used when encoding Alice's data. Can be String or Integer.          | "SuperSecretSalt1337"       |
| AliceN         | Size of N-grams used for encoding Alice's data.                                                                    | 2                           |
| AliceMetric    | Similarity metric to be computed during similarity graph generation on Alice's data.                               | "dice"                      |
| EveAlgo        | Algorithm used for encoding Eve's data. One of "BloomFilter", "TabMinHash", "TwoStepHash" or None (No Encoding).   | None                        |
| EveSecret      | Secret (seed for hash function selection/salt) used when encoding Eve's data. Can be String or Integer.            | "ATotallyDifferentString42" |
| EveN           | Size of N-grams used for encoding Eve's data.                                                                      | 2                           |
| EveMetric      | Similarity metric to be computed during similarity graph generation on Eve's data.                                 | "dice"                      |

**Additional Parameters for Bloom Filter Encoding**

| Parameter Name | Description                                                                                              | Default |
|----------------|----------------------------------------------------------------------------------------------------------|---------|
| AliceBFLength  | Length of the Bloom Filters created for Alice's data. Must be a power of 2.                              | 1024    |
| AliceBits      | Number of hash functions to populate the Bloom Filter, i.e. bits per N-Gram.                             | 10      |
| AliceDiffuse   | If True, adds diffusion layer to Bloom Filter encoding.                                                   | False   |
| AliceT         | Diffusion parameter t, i.e. number of bit positions in Alice's encodings to be XORed when creating ELDs. | 10      |
| AliceEldLength | Length of the ELD, i.e. BF with applied diffusion, for Alice's encodings.                                | 1024    |
| EveBFLength    | Length of the Bloom Filters created for Eve's data. Must be a power of 2.                                | 1024    |
| EveBits        | Number of hash functions to populate the Bloom Filter, i.e. bits per N-Gram.                             | 10      |
| EveDiffuse     | If True, adds diffusion layer to Bloom Filter encoding.                                                   | False   |
| EveT           | Diffusion parameter t, i.e. number of bit positions in Eve's encodings to be XORed when creating ELDs.   | 10      |
| EveEldLength   | Length of the ELD, i.e. BF with applied diffusion, for Eve's encodings.                                  | 1024    |

**Additional Parameters for Tabulation MinHash Encoding**

| Parameter Name | Description                                                                                                                 | Default |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|---------|
| AliceNHash     | Number of (tabulation-based) hash functions to use during MinHashing of Alice's data.                                       | 1024    |
| AliceNHashBits | Number of bits to be generated per hash function during MinHashing of Alice's data. Must be 8, 16, 32 or 64.                | 64      |
| AliceNSubKeys  | Number of sub-keys to be generated from the initial 64-bit hash during MinHashing of Alice's data. Must be a divisor of 64. | 8       |
| Alice1BitHash  | If True, applies LSB hashing, i.e. returns only the least significant bit of the MinHash results.                           | True    |
| EveNHash       | Number of (tabulation-based) hash functions to use during MinHashing of Eve's data.                                         | 1024    |
| EveNHashBits   | Number of bits to be generated per hash function during MinHashing of Eve's data. Must be 8, 16, 32 or 64.                  | 64      |
| EveNSubKeys    | Number of sub-keys to be generated from the initial 64-bit hash during MinHashing of Eve's data. Must be a divisor of 64.   | 8       |
| Eve1BitHash    | If True, applies LSB hashing, i.e. returns only the least significant bit of the MinHash results.                           | True    |

**Additional Parameters for Two-Step Hash Encoding**

| Parameter Name | Description                                                                                                                                             | Default |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| AliceNHashFunc | Number of hash functions (bits per N-gram) to use when populating intermediate BFs for Alice's data.                                                    | 10      |
| AliceNHashCol  | Number of columns (length) of intermediate BFs computed on Alice's data.                                                                                | 1000    |
| AliceRandMode  | Algorithm to be used for column-wise hashing of Alice's intermediate encodings. Either "PNG" for PRNG or "SHA" for SHA-256.                             | "PNG"   |
| EveNHashFunc   | Number of hash functions (bits per N-gram) to use when populating intermediate BFs for Eve's data.                                                      | 10      |
| EveNHashCol    | Number of columns (length) of intermediate BFs computed on Eve's data.                                                                                  | 1000    |
| EveRandMode    | Algorithm to be used for column-wise hashing of Eve's intermediate encodings. Either "PNG" for PRNG or "SHA" for SHA-256.                               | "PNG"   |

___

## Embedding Configuration
**Argument Name:** `EMB_CONFIG`

| Parameter Name  | Description                                                                                                      | Default    |
|-----------------|------------------------------------------------------------------------------------------------------------------|------------|
| AliceAlgo       | Algorithm to use for embedding Alice's data. Must be "Node2Vec" or "NetMF".                                      | "Node2Vec" |
| AliceQuantile   | Drop edges with the lowest edge weights in Alice's similarity graph. Must be >=0 (keep all) and <1.              | 0.9        |
| AliceDiscretize | If True, sets all edge weights in Alice's similarity graph to 1.                                                 | False      |
| AliceDim        | Dimensionality of Alice's embeddings.                                                                            | 128        |
| AliceContext    | Context size used when calculating Alice's embeddings.                                                           | 10         |
| AliceNegative   | Number of negative samples during training (NetMF only).                                                         | 1          |
| AliceNormalize  | If True, normalize Alice's embeddings (NetMF only).                                                              | False      |
| EveAlgo         | Algorithm to use for embedding Eve's data. Must be "Node2Vec" or "NetMF".                                        | "Node2Vec" |
| EveQuantile     | Drop edges with the lowest edge weights in Eve's similarity graph. Must be >=0 (keep all) and <1.                | 0.9        |
| EveDiscretize   | If True, sets all edge weights in Eve's similarity graph to 1.                                                   | False      |
| EveDim          | Dimensionality of Eve's embeddings.                                                                              | 128        |
| EveContext      | Context size used when calculating Eve's embeddings.                                                             | 10         |
| EveNegative     | Number of negative samples during training (NetMF only).                                                         | 1          |
| EveNormalize    | If True, normalize Eve's embeddings (NetMF only).                                                                | False      |

**Additional Parameters for Node2Vec Embedding**

| Parameter Name | Description                                                                                      | Default |
|----------------|--------------------------------------------------------------------------------------------------|---------|
| AliceWalkLen   | Length of the random walks performed on Alice's similarity graph.                                | 100     |
| AliceNWalks    | Number of random walks performed per node in Alice's similarity graph.                           | 20      |
| AliceP         | Return parameter governing random walks on Alice's similarity graph.                             | 250     |
| AliceQ         | In-Out parameter governing random walks on Alice's similarity graph.                             | 300     |
| AliceEpochs    | Number of epochs for training Alice's embeddings.                                                | 5       |
| AliceSeed      | Random seed for generating Alice's embeddings.                                                   | 42      |
| EveWalkLen     | Length of the random walks performed on Eve's similarity graph.                                  | 100     |
| EveNWalks      | Number of random walks performed per node in Eve's similarity graph.                             | 20      |
| EveP           | Return parameter governing random walks on Eve's similarity graph.                               | 250     |
| EveQ           | In-Out parameter governing random walks on Eve's similarity graph.                               | 300     |
| EveEpochs      | Number of epochs for training Eve's embeddings.                                                  | 5       |
| EveSeed        | Random seed for generating Eve's embeddings.                                                     | 42      |

___

## Alignment Configuration
**Argument Name:** `ALIGN_CONFIG`

| Parameter Name | Description                                                                                                      | Default             |
|----------------|------------------------------------------------------------------------------------------------------------------|---------------------|
| RegWS          | Regularization parameter for Sinkhorn solver.                                                                    | max(0.1, Overlap/2) |
| RegInit        | Regularization parameter for convex initialization.                                                               | 1                   |
| Batchsize      | Batch size for Wasserstein Procrustes. If <=1 interpreted as share of data, if >1 as absolute number.             | 1                   |
| LR             | Learning rate for optimization.                                                                                   | 200                 |
| LRDecay        | Learning rate decay factor per epoch.                                                                             | 1                   |
| NIterInit      | Number of iterations during convex initialization.                                                                | 5                   |
| NIterWS        | Number of iterations per optimization epoch.                                                                      | 100                 |
| NEpochWS       | Number of optimization epochs.                                                                                    | 100                 |
| Sqrt           | If True, compute alignment on the square root of embeddings.                                                     | True                |
| EarlyStopping  | Stop optimization if loss hasn’t improved for this many epochs.                                                  | 10                  |
| Selection      | Algorithm to select records for alignment ("GroundTruth", "Random", or None).                                    | None                |
| MaxLoad        | Number of records to use for alignment if Selection is not None. Must be smaller than smaller dataset size.       | None                |
| Wasserstein    | If True, uses unsupervised Wasserstein Procrustes; otherwise uses supervised closed-form Procrustes.             | True                |

___
