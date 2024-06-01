package group.vlp.lhp;

import java.io.IOException;
import javax.enterprise.context.ApplicationScoped;
import javax.faces.context.ExternalContext;
import javax.faces.context.FacesContext;
import javax.inject.Named;

@ApplicationScoped
@Named
public class MobileSwitch {

  private RenderMode findBrowserRenderMode() {
    FacesContext facesContext = FacesContext.getCurrentInstance();
    ExternalContext externalContext = facesContext.getExternalContext();
    String userAgent = externalContext.getRequestHeaderMap().get("User-Agent");
    return userAgent.toLowerCase().contains("mobile") ? RenderMode.MOBILE : RenderMode.WEB;
  }

  public String findRenderMode() {
    return findBrowserRenderMode().name();
  }

  public void switchTo(String renderMode) throws IOException {
    FacesContext facesContext = FacesContext.getCurrentInstance();
    String viewId = facesContext.getViewRoot().getViewId();
    RenderMode nextRenderMode = RenderMode.valueOf(renderMode);
    ExternalContext externalContext = facesContext.getExternalContext();
    String context = externalContext.getRequestContextPath();
    viewId = viewId.substring(viewId.lastIndexOf("/") + 1);
    String nextRenderName = nextRenderMode.name().toLowerCase();
    if (nextRenderName.equals("web"))
      nextRenderName = "";
    externalContext.redirect(String.format("%s/%s/%s", context, nextRenderName, viewId));
  }
}
