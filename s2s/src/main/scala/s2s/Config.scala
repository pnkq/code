package s2s

case class Config(
  station: String = "viet-tri",
  mode: String = "eval",
  data: String = "simple",
  lookBack: Int = 7,
  horizon: Int = 5,
  numLayers: Int = 2,
  hiddenSize: Int = 128,
  epochs: Int = 30,
  dropoutRate: Double = 0.1,
  learningRate: Double = 1E-4,
  batchSize: Int = 64,
  plot: Boolean = false,
  verbose: Boolean = false,
  save: Boolean = false,
  driverMemory: String = "16g",
  executorMemory: String = "8g",
  modelType: Int = 1, // 1=lstm, 2=lstm+bert
  bidirectional: Boolean = false
)
