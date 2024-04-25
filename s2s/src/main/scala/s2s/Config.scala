package s2s

case class Config(
  driverMemory: String = "8g",
  executorMemory: String = "4g",
  mode: String = "eval",
  modelType: String = "lstm",
  data: String = "simple",
  lookback: Int = 5,
  horizon: Int = 7,
  bidirectional: Boolean = false,
  numLayers: Int = 3,
  hiddenSize: Int = 128,
  dropoutRate: Double = 0.1,
  batchSize: Int = 64,
  learningRate: Double = 1E-4,
  epochs: Int = 30,
  station: String = "viet-tri"
)
