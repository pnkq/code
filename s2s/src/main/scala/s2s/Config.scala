package s2s

case class Config(
  driverMemory: String = "4g",
  executorMemory: String = "4g",
  mode: String = "eval",
  modelType: String = "lstm",
  data: String = "simple",
  lookback: Int = 7,
  horizon: Int = 3,
  bidirectional: Boolean = false,
  numLayers: Int = 2,
  hiddenSize: Int = 32,
  dropoutRate: Double = 0.2,
  batchSize: Int = 64,
  learningRate: Double = 1E-4,
  epochs: Int = 30
)
