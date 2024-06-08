package lhp;

import jakarta.enterprise.context.RequestScoped;
import jakarta.inject.Named;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * PhuongLH
 *
 */
@Named @RequestScoped
public class Quote {
    private String content;
    private static List<String> quotes = new ArrayList<>();
    static {
        quotes.add("Giáo dục là thứ còn lại sau khi ta quên hết những gì đã học. (B.F. Skinner)");
        quotes.add("Chúng ta học làm việc bằng cách thực hành. (Aristotle)");
        quotes.add("Trí tuệ không phải là con tàu cần bơm nhiên liệu mà là củi khô cần đốt cháy. (Plutarch)");
        quotes.add("Giáo dục là chìa khoá mở cánh cửa tự do. (George Washington Caver)");
        quotes.add("Nhiệm vụ hàng đầu của nhà giáo dục là kích thích sự sống, để nó tự do bộc lộ. (Maria Monstessori)");
        quotes.add("Học mà không có khát vọng sẽ huỷ hoại trí nhớ và không đọng lại gì. (Leonardo Da Vinci)");
        quotes.add("Thành công trong mọi sự đều cần có chuẩn bị, nếu không ắt sẽ thất bại. (Khổng Tử)");
        quotes.add("Mức độ tò mò của con người phụ thuộc vào trình độ giáo dục của họ. (Jean-Jacques Rousseau)");
        quotes.add("Hành động đến từ tư duy có chủ đích. Đó là quá trình học tự nhiên. (John Deway)");
        quotes.add("Học vấn là khả năng giải quyết các tình huống của cuộc sống. (John Hibben)");
        quotes.add("Cách làm giàu tâm trí tác động lớn đến cách làm đầy túi tiền. (B. C. Forbes)");
        quotes.add("Gạt bỏ nỗi sợ cần phải là khẩu hiệu được dán ở tất cả các cổng trường. (A. S. Neill)");
        quotes.add("Điểm yếu lớn nhất của chúng ta là dễ dàng từ bỏ. Cách tốt nhất để đạt được thành công là luôn cố gắng thêm một lần nữa. (Thomas Edison)");
        quotes.add("Nếu cơ hội không đến, hãy tự mình tạo ra cơ hội! (Jonathan Winters)");
        quotes.add("Khi cảm thấy mọi thứ đều bất lợi cho bạn, hãy nhớ rằng máy bay cất cánh ngược hướng gió chứ không phải xuôi theo chiều gió. (Henry Ford)");
        quotes.add("Bí quyết đổi mới là tập trung toàn bộ sức lực, nhưng không phải để chống lại cái cũ mà để xây dựng cái mới. (Socrates)");
        quotes.add("Một người không thể tự làm công việc của chính mình thì không thể làm công việc nhà vua giao phó. (George Savile)");
        quotes.add("Tiến hành đổi mới nghĩa là bạn sẵn sàng nghe mọi người bảo rằng bạn là kẻ dở hơi. (Larry Ellison)");
        quotes.add("Có ý tưởng thì dễ. Biến ý tưởng đó thành hiện thực mới là điều khó. (Guy Kawasaki)");
        quotes.add("Có nhiều mặt hàng được mua bằng cảm giác hơn là bằng lí trí. (George Henning)");
        quotes.add("Sự kiên nhẫn tựa như con ngựa già, mỏi mệt, nhưng vẫn sẽ cố lê bước. (William Shakespeare)");
        quotes.add("Tôi cực kỳ kiên nhẫn, miễn là cuối cùng tôi được như ý của mình. (Margaret Thatcher)");
        quotes.add("Sự kiên nhẫn và thời gian làm được nhiều điều hơn sức mạnh và niềm đam mê. (Jean de la Fontaine)");
        quotes.add("Hi vọng khởi sinh từ trong bóng tối, đó là niềm hi vọng ngoan cường rằng nếu bạn cứ hiện diện và cố gắng làm điều đúng đắn thì bình minh sẽ đến. (Anne Lamott)");
        quotes.add("Hiện tại, bạn cần xem xét sự việc. Có lẽ từ từ, vào một ngày nào đó, trong lúc vô tình không để ý, bạn sẽ tìm được câu trả lời. (Rainer Maria Rilke Patience)");
        quotes.add("Cố gắng để hiểu giống như áp lực khi đi qua vùng nước bùn. Cần kiên nhẫn chờ đợi. Hãy đứng yên và đợi bùn lắng xuống. (Lão Tử)");
        quotes.add("Những cậy chậm lớn luôn kết được quả ngọt nhất. (Molière Patience)");
        quotes.add("Giọt nước tạo nên lỗ hổng trên đá không phải nhờ sức mạnh, mà nhờ nó không ngừng rơi xuống. (Hugh Latimer)");
        quotes.add("Mọi lỗi lầm của con người đều là hậu quả của sự thiếu kiên nhẫn, làm gián đoạn quá trình có trật tự, tạo nên trở ngại hiển nhiên với một vấn đề rõ ràng. (Franz Kafka)");
        quotes.add("Lịch sử không thể vội vàng. Nhiều thế hệ các nhà cải cách đã phải trả giá để học được điều này. (Malcom Forbes)");
        quotes.add("Find out what you like doing best, and get someone to pay you for doing it. (Katharine Whitehorn)");
        quotes.add("The most common way people give up their power is by thinking they don't have any. (Alice Walker)");
        quotes.add("It does not matter how slowly you go as long as you do not stop. (Confucius)");
        quotes.add("If opportunity doesn’t knock, build a door. (Milton Berle)");
        quotes.add("Wanting to be someone else is a waste of the person you are. (Kurt Cobain)");
        quotes.add("Nothing will work unless you do. (Maya Angelou)");
        quotes.add("If you’re going through hell, keep going. (Winston Churchill)");
        quotes.add("The future depends on what you do today. (Mahatma Gandhi)");
        quotes.add("You only live once, but if you do it right, once is enough. (Mae West)");
        quotes.add("I am not a product of my circumstances. I am a product of my decisions. (Stephen Covey)");
        quotes.add("Start by doing what is necessary, then do what is possible, and suddenly you are doing the impossible. (Francis of Assisi)");
        quotes.add("Write your own book instead of reading someone else's book about success. (Herb Brooks)");
        // https://www.themuse.com/advice/50-inspirational-career-quotes
    }

    public String getContent() {
        int j = new Random().nextInt(quotes.size());
        content = quotes.get(j);
        return content;
    }
}
