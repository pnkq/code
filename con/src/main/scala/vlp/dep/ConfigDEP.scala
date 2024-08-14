package vlp.dep

case class ConfigDEP(
    master: String = "local[*]",
    driverMemory: String = "8g", // D
    mode: String = "eval",
    maxVocabSize: Int = 32768,
    tokenEmbeddingSize: Int = 32,
    tokenHiddenSize: Int = 64,
    partsOfSpeechEmbeddingSize: Int = 25,
    layers: Int = 2, // number of LSTM layers or BERT blocks
    heads: Int = 2, // number of attention heads in BERT
    batchSize: Int = 64,
    maxSeqLen: Int = 30,
    epochs: Int = 50,
    learningRate: Double = 5E-4,
    dropoutRate: Float = 0.1f,
    language: String = "vie", // [eng, ind, vie]
    modelPath: String = "bin/dep",
    outputPath: String = "out/dep/",
    scorePath: String = "dat/depx-scores-uas.tsv",
    modelType: String = "t+p", // [t+p, tg+p, tn+p, b] // [t, tg, tn]
    weightedLoss: Boolean = false,
    las: Boolean = false // labeled attachment score (LAS) or unlabeled attachment score (UAS)
)

