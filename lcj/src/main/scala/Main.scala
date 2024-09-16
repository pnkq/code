package vlp.lcj

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.{SystemMessage, UserMessage}
import java.time.LocalDate

trait Assistant:
  def chat(userMessage: String): String


trait Friend:
  @SystemMessage(Array("You are a good friend of mine. Answer using slang."))
  def chat(userMessage: String): String

trait FriendVie:
  def chat(userMessage: String): String


case class Person(firstName: String, lastName: String, birthDate: LocalDate)

trait PersonExtractor:
  @UserMessage(Array("Extract information about a person from {{it}}"))
  def extractPersonFrom(text: String): Person


@main def hello(): Unit =
  val model = OpenAiChatModel.withApiKey("demo")
  val personExtractor = AiServices.create(classOf[PersonExtractor], model)
  val text = """
           In 1968, amidst the fading echoes of Independence Day,
            a child named John arrived under the calm evening sky.
            This newborn, bearing the surname Doe, marked the start of a new journey.
            He was welcomed into the world at 345 Whispering Pines Avenue
            a quaint street nestled in the heart of Springfield
            an abode that echoed with the gentle hum of suburban dreams and aspirations.
  """
  val person = personExtractor.extractPersonFrom(text)
  print(person)
  // val assistant = AiServices.create(classOf[Assistant], model)
  // val answerFr = assistant.chat("Bonjour, enchanter de vous voir!")
  // println(answerFr)
  // val answerEn = assistant.chat("Hello, nice to meet you!")
  // println(answerEn)

  // val friend = AiServices.create(classOf[Friend], model)
  // val answer = friend.chat("Hello, how are you buddy?")
  // println(answer)

  // val friendVie = AiServices.builder(classOf[FriendVie])
  //   .chatLanguageModel(model)
  //   .systemMessageProvider(chatMemoryId => "Ông là người bạn thân của tôi. Hãy trả lời thoải mái, không cần lịch sự.")
  //   .build()

  // val answerVie = friendVie.chat("Chào, khỏe không ông?")
  // println(answer)

