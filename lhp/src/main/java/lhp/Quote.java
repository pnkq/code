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
        quotes.add("If opportunity doesn't knock, build a door. (Milton Berle)");
        quotes.add("Wanting to be someone else is a waste of the person you are. (Kurt Cobain)");
        quotes.add("Nothing will work unless you do. (Maya Angelou)");
        quotes.add("If you're going through hell, keep going. (Winston Churchill)");
        quotes.add("The future depends on what you do today. (Mahatma Gandhi)");
        quotes.add("You only live once, but if you do it right, once is enough. (Mae West)");
        quotes.add("I am not a product of my circumstances. I am a product of my decisions. (Stephen Covey)");
        quotes.add("Start by doing what is necessary, then do what is possible, and suddenly you are doing the impossible. (Francis of Assisi)");
        quotes.add("Write your own book instead of reading someone else's book about success. (Herb Brooks)");
        quotes.add("Style is knowing who you are, what you want to say, and not giving a damn. (Gore Vidal)");
        quotes.add("Getting fired is nature's way of telling you that you had the wrong job in the first place. (Hal Lancaster)");
        quotes.add("The future belongs to those who believe in the beauty of their dreams. (Eleanor Roosevelt)");
        quotes.add("The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle. (Steve Jobs)");
        quotes.add("There is no passion to be found playing small—in settling for a life that is less than the one you are capable of living. (Nelson Mandela)");
        quotes.add("Do the best you can in every task, no matter how unimportant it may seem at the time. No one learns more about a problem than the person at the bottom. (Sandra Day O'Connor)");
        quotes.add("Whatever you decide to do, make sure it makes you happy. (Paulo Coelho)");
        quotes.add("To avoid criticism, do nothing, say nothing, and be nothing. (Elbert Hubbard)");
        quotes.add("If you're offered a seat on a rocket ship, don't ask what seat! Just get on. (Sheryl Sandberg)");
        quotes.add("Be yourself; everyone else is already taken. (Oscar Wilde)");
        quotes.add("Without leaps of imagination or dreaming, we lose the excitement of possibilities. Dreaming, after all, is a form of planning. (Gloria Steinem)");
        quotes.add("A mind that is stretched by new experiences can never go back to its old dimensions. (Oliver Wendell Holmes, Jr.)");
        quotes.add("Keep away from people who try to belittle your ambitions. Small people always do that, but the really great make you feel that you, too, can become great. (Mark Twain)");
        quotes.add("He who never makes an effort, never risks a failure. (Anonymous)");
        quotes.add("When I let go of what I am, I become what I might be. (Lao Tzu)");
        quotes.add("You know, everybody has setbacks in their life, and everybody falls short of whatever goals they might set for themselves. That's part of living and coming to terms with who you are as a person. (Hillary Rodham Clinton)");
        quotes.add("If it scares you, it might be a good thing to try. (Seth Godin)");
        quotes.add("Too many of us are not living our dreams because we are living our fears. (Les Brown)");
        quotes.add("If you judge people, you have no time to love them. (Mother Teresa)");
        quotes.add("When you get into a tight place and everything goes against you until it seems that you cannot hold on for a minute longer, never give up then, for that is just the place and time when the tide will turn. (Harriet Beecher Stowe)");
        quotes.add("Make sure your own worst enemy doesn't live between your own two ears. (Laird Hamilton)");
        quotes.add("There are many things in life that will catch your eye, but only a few will catch your heart. Pursue these. (Michael Nolan)");
        quotes.add("When the grass looks greener on the other side of the fence, it may be that they take better care of it there. (Cecil Selig)");
        quotes.add("The most difficult thing is the decision to act, the rest is merely tenacity. The fears are paper tigers. You can do anything you decide to do. You can act to change and control your life; and the procedure, the process is its own reward. (Amelia Earhart)");
        quotes.add("Don't feel sorry for yourself, only assholes do that. (Haruki Murakami)");
        quotes.add("I was once afraid of people saying \"Who does she think she is?\" Now I have the courage to stand and say, \"This is who I am.\" (Oprah)");
        quotes.add("The best revenge is massive success. (Frank Sinatra)");
        quotes.add("There are no shortcuts to any place worth going. (Beverly Sills)");
        quotes.add("I am convinced that life is 10% what happens to me and 90% of how I react to it. (Charles Swindoll)");
        quotes.add("If you want to achieve greatness, stop asking for permission. (Eddie Colla)");
        quotes.add("I was always looking outside myself for strength and confidence, but it comes from within. It is there all the time. (Anna Freud)");
        quotes.add("Whatever your life's work is, do it well. A man should do his job so well that the living, the dead, and the unborn could do it no better. (Martin Luther King Jr.)");
        quotes.add("There isn't any great mystery about me. What I do is glamorous and has an awful lot of white-hot attention placed on it. But the actual work requires the same discipline and passion as any job you love doing, be it as a very good pipe fitter or a highly creative artist. (Tom Hanks)");
        quotes.add("Act as if what you do makes a difference. It does. (William James)");
        quotes.add("If you want to be a true professional, do something outside yourself. (Ruth Bader Ginsburg)");
        quotes.add("Con người khôn ngoan không nhờ kinh nghiệm mà là nhờ khả năng trải nghiệm. (George Bernard Shaw)");
        quotes.add("Live as if you were to die tomorrow. Learn as if you were to live forever. (Mahatma Gandhi)");
        quotes.add("Life isn't about finding yourself. Life is about creating yourself. (George Bernard Shaw)");
        quotes.add("The greatest lesson in life is to know that even fools are right sometimes. (Winston Churchill)");
        quotes.add("Attitude is a choice. Happiness is a choice. Optimism is a choice. Kindness is a choice. Giving is a choice. Respect is a choice. Whatever choice you make makes you. Choose wisely. (Roy T. Bennett)");
        quotes.add("Instead of worrying about what you cannot control, shift your energy to what you can create. (Roy T. Bennett)");
        quotes.add("Accept yourself, love yourself, and keep moving forward. If you want to fly, you have to give up what weighs you down. (Roy T. Bennett)");
        quotes.add("Live the Life of Your Dreams: Be brave enough to live the life of your dreams according to your vision and purpose instead of the expectations and opinions of others. (Roy T. Bennett)");
        quotes.add("I've never fooled anyone. I've let people fool themselves. They didn't bother to find out who and what I was. Instead they would invent a character for me. I wouldn't argue with them. They were obviously loving somebody I wasn't. (Marilyn Monroe)");
    }

    public String getContent() {
        int j = new Random().nextInt(quotes.size());
        content = quotes.get(j);
        return content;
    }
}
