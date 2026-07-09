import unicodedata
from p.dispatcher import Dispatcher
from tqdm import tqdm


dispatcher = Dispatcher()

with open("20231101/part-00000-b2514431-1ed1-4374-ba72-5814e6ba27cf-c000.txt", "r", encoding="utf-8") as fIn, \
     open("20231101/eng.txt", "w", encoding="utf-8") as fOut:

    progress = tqdm(unit=" spans")
    for line in fIn:
        nfc_line = unicodedata.normalize("NFC", line)
        spans = dispatcher.dispatch(nfc_line)
        words = []
        for span in spans:
            if span.lang == "eng":
                words.append(span.text)
        progress.update(len(spans))

        if words:
            fOut.write(" ".join(words))
            fOut.write("\n")
    progress.close()

            