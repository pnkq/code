package vlp.lcj

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.service.AiServices

trait Assistant:
  def chat(userMessage: String): String

@main def hello(): Unit =
  val model = OpenAiChatModel.withApiKey("demo")
  val assistant = AiServices.create(classOf[Assistant], model)
  val answerFr = assistant.chat("Bonjour, enchanter de vous voir!")
  print(answerFr)
  val answerEn = assistant.chat("Hello, nice to meet you!")
  print(answerEn)

