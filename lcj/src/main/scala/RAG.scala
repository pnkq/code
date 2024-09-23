package vlp.lcj

import dev.langchain4j.service.AiServices
import dev.langchain4j.service.{SystemMessage, UserMessage}
import dev.langchain4j.model.huggingface.HuggingFaceChatModel
import dev.langchain4j.model.huggingface.HuggingFaceModelName.TII_UAE_FALCON_7B_INSTRUCT
import dev.langchain4j.data.message.SystemMessage.systemMessage
import dev.langchain4j.data.message.UserMessage.userMessage
import java.time.Duration.ofSeconds
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor
import dev.langchain4j.memory.chat.MessageWindowChatMemory
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever
import dev.langchain4j.service.Result


trait AssistantCNN:
  def chat(userMessage: String): Result[String]

@main def rag(): Unit =
  val model = HuggingFaceChatModel.builder()
                .accessToken(System.getenv("HF_API_KEY"))
                .modelId(TII_UAE_FALCON_7B_INSTRUCT)
                .timeout(ofSeconds(15))
                .temperature(0.7)
                .maxNewTokens(20)
                .waitForModel(true)
                .build()

  val documents = FileSystemDocumentLoader.loadDocument("dat/cnn.txt")
  val embeddingStore = new InMemoryEmbeddingStore[TextSegment]()
  EmbeddingStoreIngestor.ingest(documents, embeddingStore)


  val assistant = AiServices.builder(classOf[AssistantCNN])
    .chatLanguageModel(model)
    .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
    .contentRetriever(EmbeddingStoreContentRetriever.from(embeddingStore))
    .build()

  var result = assistant.chat("How many people are killed?")
  println("CONTENT:")
  println(result.content())

  println("SOURCES:")
  var sources = result.sources()
  sources.forEach(println)
  

