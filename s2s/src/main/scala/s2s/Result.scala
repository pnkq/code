package s2s

case class ExperimentConfig(
  station: String,
  horizon: Int,
  lookBack: Int,
  layers: Int,
  hiddenSize: Int,
  epochs: Int,
  dropoutRate: Double,
  learningRate: Double,
  modelType: Int = 1
)

case class Result(
  maeU: Array[Double],
  mseU: Array[Double],
  maeV: Array[Double],
  mseV: Array[Double],
  trainingTime: Long,
  config: ExperimentConfig
)
