package group.vlp.lhp;


/**
 * Render mode on desktop web or mobile.
 */
public enum RenderMode {
  WEB("HTML_BASIC"),
  MOBILE("PRIMEFACES_MOBILE");
  private String renderKit;

  RenderMode(String renderKit) {
    this.renderKit = renderKit;
  }

  public String getRenderKit() {
    return this.renderKit;
  }

  public static RenderMode findRenderMode(String renderKit) {
    RenderMode renderMode = RenderMode.WEB;
    for (RenderMode rm : RenderMode.values()) {
      if (rm.getRenderKit().equals(renderKit)) {
        return rm;
      }
    }
    return renderMode;
  }
}