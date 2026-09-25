package tpa

import scala.io.Source
import scala.util.matching.Regex

case class MLMResult(loss: Double, top1Accuracy: Double, topkAccuracy: Double)
case class ModelAverages(loss: Double, top1Accuracy: Double, top1Std: Double, topkAccuracy: Double, topkStd: Double, sampleCount: Int)

object Scores {

  val ModelHeaderPattern: Regex = """"(\./[^"]+)"""".r
  val ResultPattern: Regex      = """MLMEvaluationResult\(loss=([\d\.]+),\s*top1_accuracy=([\d\.]+),\s*topk_accuracy=([\d\.]+).*""".r

  def main(args: Array[String]): Unit = {
    val filePath = "/home/phuonglh/code/con/src/main/python/tpa/accuracy.txt"
    val results: Map[String, ModelAverages] = parseAndAverage(filePath)

    // Output formatted results
    results.toList.sortBy(_._1).foreach { case (model, avg) =>
      println(s"Model: $model (${avg.sampleCount} samples)")
      println(f"  Loss:          ${avg.loss}%.4f")
      println(f"  Top-1 Acc:     ${avg.top1Accuracy}%.4f")
      println(f"  Top-1 Std:     ${avg.top1Std}%.6f")
      println(f"  Top-K Acc:     ${avg.topkAccuracy}%.4f")
      println(f"  Top-K Std:     ${avg.topkStd}%.6f\n")
    }
  }

  def parseAndAverage(filePath: String): Map[String, ModelAverages] = {
    val source = Source.fromFile(filePath)
    
    var currentModel: Option[String] = None
    var rawData = Vector.empty[(String, MLMResult)]

    try {
      for (line <- source.getLines()) {
        val trimmed = line.trim
        trimmed match {
          // Detect path header: e.g., "./t-model_64_4_4_256/"
          case ModelHeaderPattern(modelName) =>
            currentModel = Some(modelName)

          // Detect evaluation line and append to list
          case ResultPattern(loss, top1, topk) if currentModel.isDefined =>
            val res = MLMResult(loss.toDouble, top1.toDouble, topk.toDouble)
            rawData = rawData :+ (currentModel.get -> res)

          case _ => // Ignore empty or comment lines
        }
      }
    } finally {
      source.close()
    }

    // Group by model name and compute averages
    rawData
      .groupBy { case (modelName, _) => modelName }
      .map { case (modelName, entries) =>
        val scores = entries.map(_._2)
        val count  = scores.size
        val top1Avg = scores.map(_.top1Accuracy).sum / count
        val topkAvg = scores.map(_.topkAccuracy).sum / count
        
        val avgResult = ModelAverages(
          loss         = scores.map(_.loss).sum / count,
          top1Accuracy = top1Avg,
          top1Std = Math.sqrt(scores.map(_.top1Accuracy).map(x => (x - top1Avg)*(x - top1Avg)).sum / count),
          topkAccuracy = topkAvg,
          topkStd = Math.sqrt(scores.map(_.topkAccuracy).map(x => (x - topkAvg)*(x - topkAvg)).sum / count),
          sampleCount  = count
        )
        modelName -> avgResult
      }
  }
}

// ### Key Highlights

// * **Dynamic Header Tracking:** Replaces hardcoded strings with `ModelHeaderPattern` so every model section (like `"./t-model_64_4_4_256/"`) automatically initializes a new block.
// * **Vector Accumulation:** Collects tuple entries of `(modelName, MLMResult)` line-by-line while streaming through the file.
// * **Scala Collection Operations:** Uses `.groupBy` combined with `.map` to isolate scores per key and calculate averages cleanly.