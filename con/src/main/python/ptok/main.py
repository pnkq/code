import unicodedata
from p.tokenizer import HybridTokenizer
from p.vocabulary import Vocabulary


text = """
Sự trỗi dậy của công nghệ bảo hiểm (Insurtech) 
Hoàng An
Trong những năm vừa qua, khi thu nhập người dân được cải thiện và số hóa ngày càng được ứng dụng mạnh mẽ hơn, ngành bảo hiểm đã có nhiều sự thay đổi. Công nghệ bảo hiểm (hay còn gọi là Insurtech) nổi lên như một lĩnh vực tiềm năng và có sức tăng trưởng tốt...
The WISE Talk số thứ 06 với chủ đề: “Sự trỗi dậy của công nghệ bảo hiểm (Insurtech)” sẽ được phát sóng trực tuyến vào hồi 14:00 giờ ngày 30/3/2023 trên nền tảng VnEconomy và Fanpage VnEconomy.
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

