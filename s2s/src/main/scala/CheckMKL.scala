import com.intel.analytics.bigdl.mkl.MKL

object CheckMKL {
  def main(args: Array[String]): Unit = {
    println("Is MKL loaded? " + MKL.isMKLLoaded)
    println("MKL numThreads() = " + MKL.getNumThreads)
  }
}
