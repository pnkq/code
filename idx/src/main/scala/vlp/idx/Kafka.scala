package vlp.idx

import java.{util => ju}
import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.consumer.KafkaConsumer
import java.time.Duration

/**
  * Kafka producer and consumer for sending/receving extracted articles.
  * <p>
  * phuonglh@gmail.com, 2021
  */
object Kafka {

    val SERVERS: String = "http://112.137.134.8:9092"
    val GROUP_ID: String = "news"

  def createProducer(bootstrapServers: String): KafkaProducer[String, String] = {
      val props = new ju.Properties()
      props.setProperty("bootstrap.servers", bootstrapServers)
      props.setProperty("acks", "all")
      props.setProperty("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
      props.setProperty("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
      new KafkaProducer[String, String](props)
  }

  def consume(bootstrapServers: String, groupId: String): Unit = {
      val props = new ju.Properties()
      props.setProperty("bootstrap.servers", bootstrapServers)
      // set new groupId to read from the beginning
      props.setProperty("group.id", groupId + "-consumer-" + java.util.UUID.randomUUID().toString)
      props.setProperty("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
      props.setProperty("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
      props.setProperty("enable.auto.commit", "true");
      props.setProperty("auto.commit.interval.ms", "1000")
      props.setProperty("auto.offset.reset", "earliest")
      val consumer = new KafkaConsumer[String, String](props)
      consumer.subscribe(ju.Arrays.asList(groupId))
      try {
          while (true) {
              val records = consumer.poll(Duration.ofMillis(100))
              val recordIterator = records.iterator()
              while (recordIterator.hasNext) {
                val record = recordIterator.next()
                println(s"ofsset = ${record.offset}, key = ${record.key}")
              }
          }
      } finally {
          consumer.close()
      }
  }

  def main(args: Array[String]): Unit = {
    if (args.length > 0)
        consume(Kafka.SERVERS, args(0))
    else consume(Kafka.SERVERS, Kafka.GROUP_ID)
  }
}
