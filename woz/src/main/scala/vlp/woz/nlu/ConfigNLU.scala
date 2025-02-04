package vlp.woz.nlu

case class ConfigNLU(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  maxSeqLen: Int = 25,
  batchSize: Int = 32,
  learningRate: Double = 1E-4,
  embeddingSize: Int = 200,
  recurrentSize: Int = 128, // x2 for bidirectional
  numLayers: Int = 2,
  numHeads: Int = 4,
  hiddenSize: Int = 64,
  modelType: String = "lstm",
  mode: String = "eval",
  epochs: Int = 200,
  lambdaSlot: Float = 0.8f,
  embeddingType: String = "b" // d=DeBERTa, b=BERT, x=XLM-RoBERTa
)
