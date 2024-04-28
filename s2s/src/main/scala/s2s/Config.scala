package s2s

case class Config(
  driverMemory: String = "8g",
  executorMemory: String = "4g",
  mode: String = "eval",
  modelType: String = "lstm",
  data: String = "simple",
  lookback: Int = 7,
  horizon: Int = 5,
  bidirectional: Boolean = false,
  numLayers: Int = 2,
  hiddenSize: Int = 128,
  dropoutRate: Double = 0.1,
  batchSize: Int = 64,
  learningRate: Double = 1E-4,
  epochs: Int = 30,
  station: String = "viet-tri",
  plot: Boolean = false,
  verbose: Boolean = false,
  save: Boolean = false
)
