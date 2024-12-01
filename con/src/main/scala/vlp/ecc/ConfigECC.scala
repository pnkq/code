package vlp.ecc

case class ConfigECC(
  mode: String = "eval",
  batchSize: Int = 32,
  hiddenSizes: String = "", // comma separated integers, i.e., "64,32"
  epochs: Int = 20,
  learningRate: Double = 2E-5,
  modelPath: String = "bin/ecc",
  trainPath: String = "dat/ecc/ECC-train",
  validPath: String = "dat/ecc/ECC-val",
  outputPath: String = "dat/ecc/",
  scorePath: String = "dat/med/scores-ecc.json",
  modelType: String = "r"
)

