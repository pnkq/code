package vlp.dep;

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.scene.Scene;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;

public class DFX extends Application {
  public void start(Stage primaryStage) {
      Button ok = new Button();
      ok.setText("OK");
      ok.setOnAction(new EventHandler<ActionEvent>() {
         public void handle(ActionEvent event) {
             System.out.println("HW");
         }
      });
      StackPane root = new StackPane();
      root.getChildren().add(ok);
      primaryStage.setTitle("DFX");
      Scene scene = new Scene(root, 300, 250);
      primaryStage.setScene(scene);
      primaryStage.show();
  }

    public static void main(String[] args) {
        launch(args);
    }
}
