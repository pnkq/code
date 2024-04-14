package vlp.idx;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpHost;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.bulk.BulkRequest;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.common.unit.TimeValue;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.sort.SortOrder;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static org.elasticsearch.common.xcontent.XContentFactory.jsonBuilder;


class NewsList {
  List<News> news;
  public List<News> getNews() {
    return news;
  }
  public void setNews(List<News> news) {
    this.news = news;
  }
}

/**
 * Elastic Search Indexer.
 * <p/>
 * phuonglh
 */
public class Indexer {
  static RestHighLevelClient client = new RestHighLevelClient(RestClient.builder(new HttpHost("112.137.134.8", 9200, "http")));
  static final Logger logger = Logger.getLogger(Indexer.class.getName());

  /**
   * Index one news sample.
   * @param news a news
   * @throws IOException
   */
  public static void indexOne(News news) throws IOException {
    XContentBuilder builder = jsonBuilder().startObject();
    builder.field("url", news.getUrl()).field("content", news.getContent()).field("date", news.getDate());
    builder.endObject();
    IndexRequest request = new IndexRequest("news").source(builder).timeout(TimeValue.timeValueSeconds(1));
    client.indexAsync(request, RequestOptions.DEFAULT, new ActionListener<IndexResponse>() {
      @Override
      public void onResponse(IndexResponse indexResponse) {
        logger.info("Success indexing: " + news.getUrl());
      }
      @Override
      public void onFailure(Exception e) {
        logger.info("Failure indexing: " + news.getUrl());
      }
    });
  }

  /**
   * Index many news samples in one request.
   * @param news a list of texts
   * @throws IOException
   */
  public static void indexManyNews(List<News> news) throws IOException {
    BulkRequest request = new BulkRequest();
    for (News ns: news) {
      XContentBuilder builder = jsonBuilder().startObject();
      builder.field("url", ns.getUrl()).field("content", ns.getContent()).field("date", ns.getDate());
      builder.endObject();
      request.add(new IndexRequest("news").source(builder));
    }
    client.bulkAsync(request, RequestOptions.DEFAULT, new ActionListener<BulkResponse>() {
      @Override
      public void onResponse(BulkResponse bulkItemResponses) {
        logger.info("Success in indexing " + news.size() + " news.");
      }
      @Override
      public void onFailure(Exception e) {
        e.printStackTrace();
        logger.info("Failure in indexing " + news.size() + " news.");
      }
    });
  }

  /**
   * Index all news samples from a JSON input file Note that
   * we need to filter existing URLs from the index.
   * @param jsonInputPath an input file in JSON format.
   * @throws IOException
   */
  public static void indexManyNews(String jsonInputPath) throws IOException {
    ObjectMapper objectMapper = new ObjectMapper();
    NewsList list = new NewsList();
    try {
      InputStream inputStream = new FileInputStream(jsonInputPath);
      list = objectMapper.readValue(inputStream, NewsList.class);
      inputStream.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    System.out.println("Number of news = " + list.news.size());

    // load the news index and build a set of existing URLs
    Set<String> existingURLs = new HashSet<>();
    // create a request
    SearchRequest request = new SearchRequest("news");
    SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
    searchSourceBuilder.query(QueryBuilders.matchAllQuery());
    searchSourceBuilder.size(10000);
//    searchSourceBuilder.sort("date", SortOrder.DESC);
    request.source(searchSourceBuilder);
    // execute the request and get a response
    SearchResponse response = client.search(request, RequestOptions.DEFAULT);
    // extract and return result
    SearchHit[] hits = response.getHits().getHits();
    for (SearchHit hit: hits) {
      String url = hit.getSourceAsMap().get("url").toString();
      existingURLs.add(url);
    }
    // filter novel news
    List<News> novelNews = list.news.stream().filter(element -> !existingURLs.contains(element.getUrl()))
        .filter(element -> element.getContent().trim().length() >= 200 & !element.getContent().contains("<div") &&
            !element.getContent().contains("<table") && !element.getContent().contains("</p>"))
        .collect(Collectors.toList());

    logger.info("#(novelNews) = " + novelNews.size());
    // divide novel news into small chunks of 2000 samples
    int batch = 2000;
    int n = novelNews.size() / batch;
    for (int i = 0; i < n; i++) {
      indexManyNews(novelNews.subList(batch*i, batch*(i+1)));
      try {
        Thread.sleep(5000);
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
      logger.info("indexed: " + batch*(i+1));
    }
    indexManyNews(novelNews.subList(n*batch, novelNews.size()));
  }

  public static void close() {
    try {
      if (client != null)
        client.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws IOException, InterruptedException {
    indexManyNews("dat/20240116.json");
    Thread.sleep(5000);
    Indexer.close();
    System.out.println("Done.");
  }
}
