package s2s

import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot.displayPlot
import com.cibo.evilplot.plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.colors._
import com.cibo.evilplot.plot.renderers.PathRenderer
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.sql.{DataFrame, SparkSession}

object Plot {
  def plot(spark: SparkSession, config: Config, prediction: DataFrame): Unit = {
    // draw multiple plots corresponding to number of time steps up to horizon for the first year of the validation set
    import spark.implicits._
    val ys = prediction.select("label").map(row => row.getAs[Vector](0).toDense.toArray).take(365).zipWithIndex
    val zs = prediction.select("prediction").map(row => row.getAs[Seq[Float]](0)).take(365).zipWithIndex

    // plot +1 day prediction and label
    val dataY0 = ys.map(pair => Point(pair._2, pair._1.head))
    val dataZ0 = zs.map(pair => Point(pair._2, pair._1.head))
    displayPlot(Overlay(
      LinePlot(dataY0, Some(PathRenderer.named[Point](name = "horizon=0", strokeWidth = Some(1.2), color = HTMLNamedColors.gray))),
      LinePlot(dataZ0, Some(PathRenderer.default[Point](strokeWidth = Some(1.2), color = Some(HTMLNamedColors.blue))))
    ).xAxis().xLabel("day").yAxis().yLabel("rainfall").yGrid().bottomLegend())

    val plots = (0 until config.horizon).map { d =>
      val dataY = ys.map(pair => Point(pair._2, pair._1(d)))
      val dataZ = zs.map(pair => Point(pair._2, pair._1(d)))
      Seq(
        LinePlot(dataY, Some(PathRenderer.named[Point](name = s"horizon=$d", strokeWidth = Some(1.2), color = HTMLNamedColors.gray)))
          .topPlot(LinePlot(dataZ, Some(PathRenderer.default[Point](strokeWidth = Some(1.2), color = Some(HTMLNamedColors.blue)))))
      )
    }
    displayPlot(Facets(plots).standard().title("Rainfall in a Year").topLegend())

    // overlay plots
    val colors = Color.getGradientSeq(config.horizon)
    val days = 0 until config.horizon
    val overlayPlots = days.map { d =>
      val dataZ = zs.map(pair => Point(pair._2, pair._1(d)))
      LinePlot(data = dataZ, Some(PathRenderer.named[Point](name = s"horizon=$d", strokeWidth = Some(1.2), color = colors(d))))
    }
    displayPlot(Overlay(overlayPlots: _*).xAxis().xLabel("day").yAxis().yLabel("rainfall").yGrid().bottomLegend())
  }

}
