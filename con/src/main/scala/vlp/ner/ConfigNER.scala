package vlp.ner

case class ConfigNER(
  master: String = "local[*]",
  totalCores: Int = 8,    // X
  executorCores: Int = 8, // Y
  executorMemory: String = "8g", // Z
  driverMemory: String = "16g", // D
  mode: String = "eval",
  batchSize: Int = 32,
  minSeqLen: Int = 10,
  maxSeqLen: Int = 80,
  layers: Int = 2, // number of LSTM layers or BERT blocks
  hiddenSize: Int = 64,
  epochs: Int = 20,
  learningRate: Double = 1E-4,
  modelPath: String = "bin/med",
  trainPath: String = "dat/med/syll.txt",
  validPath: String = "dat/med/val/", // Parquet file of devPath
  outputPath: String = "dat/med/",
  scorePath: String = "dat/med/scores-med.json",
  modelType: String = "d",
  firstTime: Boolean = false
)

