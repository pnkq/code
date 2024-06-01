package group.vlp.lhp;


import javax.annotation.PostConstruct;
import javax.enterprise.context.SessionScoped;
import javax.inject.Named;
import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.UriBuilder;
import java.io.Serializable;
import java.io.StringReader;
import java.net.URI;


@Named("quo")
@SessionScoped
public class QuoteController implements Serializable {
  private static URI uri = UriBuilder.fromUri("http://vlp.group:8088/jobs?appName=vlp&classPath=group.vlp.quo.QuoteJob&context=vlp&sync=true").build();
  private Client client = ClientBuilder.newClient();
  private WebTarget target = client.target(uri);

  private String quote = "";

  @PostConstruct
  public void initialize() {
    Response response = target.request().post(Entity.entity("", MediaType.TEXT_PLAIN_TYPE.withCharset("UTF-8")));
    String r = response.readEntity(String.class);
    JsonReader reader = Json.createReader(new StringReader(r));
    JsonObject object = reader.readObject();
    JsonArray answer = object.getJsonArray("result");
    quote = answer.getString(0) + " (" + answer.getString(1) + ")";
  }

  public String getQuote() {
    return quote;
  }

  public void setQuote(String quote) {
    this.quote = quote;
  }
}