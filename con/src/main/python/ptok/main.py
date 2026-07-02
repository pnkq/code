import unicodedata
from p.tokenizer import HybridTokenizer
from p.vocabulary import Vocabulary


text = """
Sự trỗi dậy của công nghệ bảo hiểm (Insurtech) 
Hoàng An
Trong những năm vừa qua, khi thu nhập người dân được cải thiện và số hóa ngày càng được ứng dụng mạnh mẽ hơn, ngành bảo hiểm đã có nhiều sự thay đổi. Công nghệ bảo hiểm (hay còn gọi là Insurtech) nổi lên như một lĩnh vực tiềm năng và có sức tăng trưởng tốt...
The WISE Talk số thứ 06 với chủ đề: “Sự trỗi dậy của công nghệ bảo hiểm (Insurtech)” sẽ được phát sóng trực tuyến vào hồi 14:00 giờ ngày 30/3/2023 trên nền tảng VnEconomy và Fanpage VnEconomy.
Theo báo cáo của Google và Bain năm 2022, Insurtech của Việt Nam phát triển gần 50% mỗi năm, cùng tốc độ với Indonesia và Philippines. Về quy mô tuyệt đối vẫn còn nhỏ trong khu vực, tuy nhiên nếu giữ đà tăng trưởng như hiện tại thì 6 năm sau sẽ đạt quy mô insurtech như Thailand và Singapore bây giờ. 
Tuy nhiên, thị trường bảo hiểm cũng giống như các lĩnh vực khác đều tồn tại các bài toán lớn và thách thức cần phải giải. Liệu những công nghệ mới có thực sự như một chiếc “chìa khóa vạn năng” để giúp các doanh nghiệp bảo hiểm tối ưu hóa hoạt động kinh doanh cũng như giải các bài toán về sản phẩm, phát triển thị trường và đáp ứng ngày càng tốt hơn và hiệu quả hơn các nhu cầu về bảo hiểm của người dân và doanh nghiệp...?
Xu hướng phát triển của thị trường Insurtech nói riêng tại Việt Nam trong những năm qua, đặc biệt là với sự hiện diện và tác động của đại dịch Covid-19 đã và đang diễn ra như thế nào? Những sản phẩm bảo hiểm mới trên thị trường với sự trợ lực từ công nghệ như bảo hiểm nhúng đang được phát triển và thu hút khách hàng ra sao? Về mặt quản lý, việc giám sát các mô hình kinh doanh mới, sản phẩm mới được tính đến thế nào?
Để mang tới cho độc giả những cái nhìn cận cảnh hơn về lĩnh vực này, Trung tâm Kết nối Công nghệ Tương lai (Tech Connect), Tạp chí Kinh tế Việt Nam/VnEconomy tổ chức Talkshow The WISE Talk số 06 với chủ đề: “Sự trỗi dậy của công nghệ bảo hiểm (Insurtech)” 
Các nội dung được thảo luận trong chương trình bao gồm:
Trong ngành công nghiệp bảo hiểm, công nghệ đang được ứng dụng ở mức nào? Chiến lược chuyển đổi số của các doanh nghiệp bảo hiểm hiện nay đã và đang mang lại những thay đổi đáng kể ra sao? 
Xu hướng phát triển của thị trường Insurtech nói riêng tại Việt Nam trong những năm qua, đặc biệt là với sự hiện diện và tác động của đại dịch Covid-19? 
Thị trường bảo hiểm nói chung còn tồn tại những rào cản lớn nào để phát triển và công nghệ đang giúp tháo gỡ các thách thức ấy như thế nào?
Công nghệ liệu sẽ giúp giải bài toán về niềm tin người tiêu dùng như thế nào? 
Tại Việt Nam, bảo hiểm nhúng đang được phát triển ra sao? Các câu chuyện thực tế về việc phát triển các sản phẩm bảo hiểm nhúng của doanh nghiệp tại Việt Nam
Việt Nam đã có những cơ chế chính sách nào để hỗ trợ cho sự phát triển của công nghệ bảo hiểm nói chung và bảo hiểm nhúng nói riêng? Còn những vướng mắc gì về chính sách để thị trường này phát triển lành mạnh hơn nữa?
Xu hướng phát triển của Insurtech nói chung và bảo hiểm nhúng nói riêng tại Việt Nam trong năm 2023 và 2-3 năm tới?
Khách mời của Talkshow bao gồm:
Ông Ngô Trung Dũng, Phó Tổng thư ký Hiệp hội bảo hiểm Việt Nam
Bà Lê Thị Thanh Vân, đồng sáng lập kiêm CEO Saladin, nền tảng bảo hiểm công nghệ đa giải pháp, giúp kết nối trực tuyến khách hàng với những giải pháp bảo hiểm phù hợp từ các nhà cung cấp bảo hiểm hàng đầu Việt Nam.
Host: Đào Huyền Trang  
The WISE Talk số thứ 06 với chủ đề: “Sự trỗi dậy của công nghệ bảo hiểm (Insurtech)” sẽ được phát sóng trực tuyến vào hồi 14:00 giờ ngày 30/3/2023 trên nền tảng VnEconomy và Fanpage VnEconomy.
Trân trọng kính mời quý độc giả quan tâm theo dõi!
"""

tokenizer = HybridTokenizer()
nfc_text = unicodedata.normalize("NFC", text)
pieces = tokenizer.tokenize(nfc_text)

print("TOKENS")
for p in pieces:
    print(p)


# vocab = Vocabulary()
# ids = vocab.encode(pieces)
# print(ids)

