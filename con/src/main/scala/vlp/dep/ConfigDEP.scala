package vlp.dep

case class ConfigDEP(
    master: String = "local[*]", // M
    driverMemory: String = "16g", // D
    executorMemory: String = "16g", // E
    mode: String = "eval",
    maxVocabSize: Int = 32768,
    tokenEmbeddingSize: Int = 32,
    tokenHiddenSize: Int = 64, // recurrent size or 
    partsOfSpeechEmbeddingSize: Int = 25,
    featureStructureEmbeddingSize: Int = 25,
    layers: Int = 2, // number of LSTM layers or BERT blocks
    heads: Int = 2, // number of attention heads in BERT
    batchSize: Int = 64,
    maxSeqLen: Int = 20, // max 20 tokens for short sentences
    epochs: Int = 20,
    learningRate: Double = 5E-4,
    dropoutRate: Float = 0.2f,
    language: String = "vie", // [eng, ind, vie]
    modelPath: String = "bin/dep",
    outputPath: String = "out/dep/",
    scorePath: String = "dat/depx-scores.tsv",
    modelType: String = "t", // [t+p, tg+p, tn+p, b, x, bx] // [t, tg, tn]
    weightedLoss: Boolean = false,
    las: Boolean = false // labeled attachment score (LAS) or unlabeled attachment score (UAS)
)

