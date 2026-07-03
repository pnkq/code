from tokenizers import Tokenizer

tok = Tokenizer.from_file("bpe_eng.json")

text = "transformers OpenAI uses Vietcombank and FPT for investment"

enc = tok.encode(text)

print(enc.tokens)
print(enc.ids)