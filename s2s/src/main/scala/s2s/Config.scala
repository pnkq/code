package s2s

case class Config(
  station: String = "viet-tri",
  mode: String = "eval",
  data: String = "clusterC", // {simple, complex, clusterC}
  lookBack: Int = 7,
  horizon: Int = 7,
  epochs: Int = 20,
  learningRate: Double = 1E-4,
  batchSize: Int = 64,
  plot: Boolean = false,
  verbose: Boolean = false,
  save: Boolean = false,
  master: String = "local[*]",
  driverMemory: String = "24g",
  executorMemory: String = "8g",
  modelType: Int = 1, // 1=lstm, 2=lstm+bert
  // LSTM params
  nLayer: Int = 2,
  hiddenSize: Int = 64,
  dropoutRate: Double = 0.1,
  bidirectional: Boolean = false,
  // BERT params
  nBlock: Int = 2,
  nHead: Int = 2,
  bertSize: Int = 32,
  intermediateSize: Int = 16,
  minLoss: Float = 0.2f // minLoss for early stopping
)
