package vlp.dsl

case class Config(
  language: String = "EN",
  driverMemory: String = "8g",
  batchSize: Int = 32,
  layers: Int = 2,
  maxSeqLen: Int = 40,
  tokenEmbeddingSize: Int = 100,
  tokenHiddenSize: Int = 200,
  denseHiddenSize: Int = 32,
  epochs: Int = 43,
  mode: String = "eval",
  modelType: String = "rnn",
  modelPath: String = "bin",
  outputPath: String = "out",
  scorePath: String = "out/scores.csv",
  vocabPath: String = "dat",
  weightedLoss: Boolean = false,
  pretrained: Boolean = false,
  testMode: Boolean = false
)
