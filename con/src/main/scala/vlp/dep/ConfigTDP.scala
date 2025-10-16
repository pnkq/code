package vlp.dep

/**
  * Created by phuonglh on 6/30/17.
  */
case class ConfigTDP
(
  master: String = "local[*]",
  driverMemory: String = "16g",
  mode: String = "eval",
  verbose: Boolean = false,
  modelPath: String = "bin/tdp/",
  language: String = "eng",
  classifier: String = "mlr",
  minFrequency: Int = 1,
  numFeatures: Int = 65536,
  iterations: Int = 400,
  batchSize: Int = 16,
  extended: Boolean = false, // use super-tag features
  tagEmbeddingSize: Int = 10, // super-tag embedding size
  hiddenUnits: String = "16",
  discrete: Boolean = true // use discrete super-tag features  
)