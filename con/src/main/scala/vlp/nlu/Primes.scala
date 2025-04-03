object Primes {
  def main(args: Array[String]): Unit  = {
    def isPrime(k: Int) = (k >= 2 ) && !(2 to Math.sqrt(k-1).toInt).exists(a => k % a == 0)

    (1 to 1000).filter(isPrime(_)).foreach(println)
  }
}
