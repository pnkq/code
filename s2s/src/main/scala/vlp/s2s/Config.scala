package vlp.s2s

case class Config(
  modelType: String = "lstm",
  lookback: Int = 7,
  horizon: Int = 3,
  numLayers: Int = 2,
  hiddenSize: Int = 16,
  batchSize: Int = 64,
  learningRate: Double = 1E-4,
  epochs: Int = 20
)
