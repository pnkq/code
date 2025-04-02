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
  recurrentSize: Int = 256, // x2 for bidirectional RNN
  numLayers: Int = 1,
  numHeads: Int = 4,
  hiddenSize: Int = 128, // used in transformer and BERT
  modelType: String = "lstm", // [lstm, tran, bert, join]
  mode: String = "eval",
  epochs: Int = 40,
  lambdaSlot: Float = 0.8f,
  embeddingType: String = "b", // for trainJSL mode only, [d=DeBERTa, b=BERT, x=XLM-RoBERTa]
  save: Boolean = false
)
