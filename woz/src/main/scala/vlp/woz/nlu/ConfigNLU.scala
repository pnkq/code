package vlp.woz.nlu

case class ConfigNLU(
  maxSeqLen: Int = 80,
  embeddingSize: Int = 32,
  recurrentSize: Int = 128,
  numLayers: Int = 2,
  hiddenSize: Int = 64,
  modelType: String = "lstm"
)
