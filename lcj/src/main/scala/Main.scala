package vlp.lcj

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.SystemMessage
// import dev.langchain4j.data.message.SystemMessage

trait Assistant:
  def chat(userMessage: String): String


trait Friend:
  @SystemMessage(Array("You are a good friend of mine. Answer using slang."))
  def chat(userMessage: String): String

trait FriendVie:
  def chat(userMessage: String): String

@main def hello(): Unit =
  val model = OpenAiChatModel.withApiKey("demo")
  val assistant = AiServices.create(classOf[Assistant], model)
  val answerFr = assistant.chat("Bonjour, enchanter de vous voir!")
  println(answerFr)
  val answerEn = assistant.chat("Hello, nice to meet you!")
  println(answerEn)

  val friend = AiServices.create(classOf[Friend], model)
  val answer = friend.chat("Hello, how are you buddy?")
  println(answer)

  val friendVie = AiServices.builder(classOf[FriendVie])
    .chatLanguageModel(model)
    .systemMessageProvider(chatMemoryId => "Ông là người bạn thân của tôi. Hãy trả lời thoải mái, không cần lịch sự.")
    .build()

  val answerVie = friendVie.chat("Chào, khỏe không ông?")
  println(answer)

