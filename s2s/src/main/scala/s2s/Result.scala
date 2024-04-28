package s2s

case class Result(
  maeU: Array[Double],
  mseU: Array[Double],
  maeV: Array[Double],
  mseV: Array[Double],
  trainingTime: Long,
  config: Config,
)
