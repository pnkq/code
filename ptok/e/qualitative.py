from transformers import pipeline

# Eval: fill mask probing

# fill_mask = pipeline("fill-mask", model="roberta-base")
fill_mask = pipeline("fill-mask", model="vinai/phobert-base-v2")

def predict(masked_text):
    predictions = fill_mask(masked_text)
    for p in predictions:
        print(p["token_str"], p["score"])
    for p in predictions:
        print(p)    
    print()

def predict_multiple(masked_text):
    results = fill_mask(masked_text)
    # Iterate through the predictions for each mask position
    for i, mask_predictions in enumerate(results):
        print(f"--- Top predictions for <mask> #{i + 1} ---")
        for pred in mask_predictions:
            token_str = pred["token_str"]
            score = pred["score"]
            completed_sentence = pred["sequence"]
            print(
                f"Token: '{token_str}' | Score: {score:.4f} | Full Text: {completed_sentence}"
            )        

# print("==single==")
# predict("Hanoi is the capital of <mask>.")
# predict("The keys to the cabinet <mask> on the table.")
# predict("She sat by the river <mask>.")
# predict("He deposited money at the <mask>.")
# predict("In the summer, ice cream melts because of the <mask>.")
# print("==multiple==")
# predict_multiple("The <mask> sat on the <mask>.")
# predict_multiple("<mask> created SpaceX and <mask>. <mask> is a billionaire.")

predict("Hà_Nội là thủ_đô của <mask>.")
predict("Nam lái xe chở Hoa về nhà <mask> ấy. ")
predict("Hắn đá quả <mask> bay ra xa.")
predict("Thanh_niên dùng nhiều <mask> cười để tạo ảo_giác.")
predict("Vào mùa hè, kem tan chảy nếu để ngoài trời vì <mask>.")
