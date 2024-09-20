package vlp.lcj

import dev.langchain4j.service.AiServices
import dev.langchain4j.service.{SystemMessage, UserMessage}
import dev.langchain4j.model.huggingface.HuggingFaceChatModel
import dev.langchain4j.model.huggingface.HuggingFaceModelName.TII_UAE_FALCON_7B_INSTRUCT
import dev.langchain4j.data.message.SystemMessage.systemMessage
import dev.langchain4j.data.message.UserMessage.userMessage
import java.time.Duration.ofSeconds


@main def hf(): Unit =
  val model = HuggingFaceChatModel.builder()
                .accessToken(System.getenv("HF_API_KEY"))
                .modelId(TII_UAE_FALCON_7B_INSTRUCT)
                .timeout(ofSeconds(15))
                .temperature(0.7)
                .maxNewTokens(20)
                .waitForModel(true)
                .build()
  // val aiMessage = model.generate(
  //   systemMessage("You are a data scientist. You are insterested in climate modeling. "),
  //   userMessage("Tell me something about the climate change and its impract."))

  val aiMessage = model.generate(
    systemMessage("""In 1968, amidst the fading echoes of Independence Day,
            a child named John arrived under the calm evening sky.
            This newborn, bearing the surname Doe, marked the start of a new journey.
            He was welcomed into the world at 345 Whispering Pines Avenue
            a quaint street nestled in the heart of Springfield
            an abode that echoed with the gentle hum of suburban dreams and aspirations."""),
    userMessage("Who is mentioned in the text? What places are mentionned?")
  )
  print(aiMessage.content())

