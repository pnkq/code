package vlp.dep;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import javafx.scene.control.Button;
import javafx.scene.Scene;

import java.util.ArrayList;
import java.util.List;


/**
 * phuonglh
 */
public class DFX extends Application {

    private List<String> readGraphs(String path) {
        List<String> list = new ArrayList<>();
        GraphReader.read(path).take(3).foreach(g -> list.add(g.toText()));
        return list;
    }

    public void start(Stage primaryStage) {
        GridPane topPane = new GridPane();
        topPane.setAlignment(Pos.CENTER);
        topPane.setVgap(4);
        topPane.setHgap(10);
        topPane.setPadding(new Insets(5, 5, 5, 5));
        Label label = new Label("Graph sample: ");
        ComboBox<String> comboBox = new ComboBox<>();
        List<String> xs = readGraphs("dat/dep/vi_vtb-ud-dev.conllu");
        comboBox.getItems().addAll(xs);
        comboBox.setPrefWidth(500);
        topPane.add(label, 0, 0);
        topPane.add(comboBox, 1, 0);
        Button ok = new Button();
        ok.setText("OK");
        ok.setOnAction(event -> System.out.println("HW"));
        topPane.add(ok, 2, 0);

        BorderPane centerPane = new BorderPane();

        BorderPane root = new BorderPane();
        root.setTop(topPane);
        root.setCenter(centerPane);
        Scene scene = new Scene(root, 400, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
