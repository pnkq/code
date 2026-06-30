from p.tokenizer import HybridTokenizer
from p.vocabulary import Vocabulary


text = """
Tôi đang nghiên cứu tokenizer and transformer models của OpenAI tại Hà Nội và Đà Nẵng. 
Tôi ngồi tại bán đảo Sơn Trà. 
Hôm nay là ngày 30/06/2026.
Văn bản có tiếng Anh và tiếng Việt xen lẫn nhau.
The smell of death lingers in Venezuela’s ruined streets as overwhelmed hospitals race to save lives.
"""


tokenizer = HybridTokenizer()

pieces = tokenizer.tokenize(text)

print()

print("TOKENS")

print("----------------")

for p in pieces:
    print(p)


vocab = Vocabulary()

ids = vocab.encode(pieces)

print()

print("IDS")

print("----------------")

print(ids)