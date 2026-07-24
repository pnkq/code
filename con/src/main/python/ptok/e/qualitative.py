from transformers import pipeline

fill_mask = pipeline("fill-mask", model="roberta-base")

predictions = fill_mask("Paris is the capital of <mask>.")

for p in predictions:
    print(p["token_str"], p["score"])

print()

for p in predictions:
    print(p)    

