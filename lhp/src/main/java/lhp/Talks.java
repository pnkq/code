package lhp;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.RequestScoped;
import jakarta.inject.Inject;
import jakarta.inject.Named;

import java.util.List;

@Named @RequestScoped
public class TalkController {
    private List<Talk> talks;
    @Inject
    private TalkService service;

    @PostConstruct
    public void init() {
        talks = service.list();
    }

    public List<Talk> getTalks() {
        // create some talks
        Talk talk = null;
        talk = new Talk();
        talk.setTitle("Artificial Intelligence Beyond Text, Speech and Image Processing");
        service.add(talk);
        talk = new Talk();
        talk.setTitle("Introduction to Artificial Intelligence, Machine Learning, Deep Learning and Foundation Models");
        service.add(talk);
        return talks;
    }
}
