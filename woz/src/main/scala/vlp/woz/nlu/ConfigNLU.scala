package vlp.woz.nlu

case class ConfigNLU(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  maxSeqLen: Int = 20,
  batchSize: Int = 32,
  learningRate: Double = 1E-4,
  embeddingSize: Int = 32,
  recurrentSize: Int = 64, // x2 for bidirectional
  numLayers: Int = 1,
  hiddenSize: Int = 64,
  modelType: String = "lstm",
  mode: String = "train",
  epochs: Int = 100
)
