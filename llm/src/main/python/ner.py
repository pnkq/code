import torch
import transformers

# model = transformers.AutoModel.from_pretrained('numind/NuNER-v2.0')
# tokenizer = transformers.AutoTokenizer.from_pretrained('numind/NuNER-v2.0')

# text = [
#     "NuMind is an AI company based in Paris and USA.",
#     "See other models from us on https://huggingface.co/numind",
#     "FPT là một công ty tin học ở Việt Nam."
# ]
# encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# output = model(**encoded_input)

# emb = output.last_hidden_state
# print(emb)


from gliner import GLiNER

def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged


model = GLiNER.from_pretrained("numind/NuNerZero")

# NuZero requires labels to be lower-cased!
labels = ["person", "organization", "initiative", "project", "number", "percentage"]
labels = [l.lower() for l in labels]

text = """
    At the annual technology summit, the keynote address was delivered by a senior member of the Association for Computing Machinery Special Interest Group on Algorithms and Computation Theory, which recently launched an expansive initiative titled 'Quantum Computing and Algorithmic Innovations: Shaping the Future of Technology'. This initiative explores the implications of quantum mechanics on next-generation computing and algorithm design and is part of a broader effort that includes the 'Global Computational Science Advancement Project'. The latter focuses on enhancing computational methodologies across scientific disciplines, aiming to set new benchmarks in computational efficiency and accuracy.
    Đại biểu Nguyễn Hoàng Bảo Trân (Phó chủ tịch Liên đoàn Lao động tỉnh Bình Dương) băn khoăn việc lao động phải đóng bù tiền để hưởng quyền lợi mà họ đương nhiên được hưởng. Khi doanh nghiệp chậm, trốn đóng BHXH, người lao động đã mất trắng khoản lương hàng tháng trích đóng ban đầu. Nếu phải đóng bù lần nữa cho phần doanh nghiệp vi phạm thì lao động phải đóng nhiều lần với tỷ lệ lớn (gồm 8% tiền lương tháng lần đầu, 22% khoản đóng bù lần hai).
    Đại biểu Nguyễn Thành Nam (Phó đoàn Phú Thọ) đề nghị mở rộng nguồn lực để hỗ trợ lao động yếu thế trong khoảng thời gian bị công ty chậm, trốn đóng mà chưa xử lý được.
"""

entities = model.predict_entities(text, labels)

entities = merge_entities(entities)

for entity in entities:
    print(entity["text"], "=>", entity["label"])


