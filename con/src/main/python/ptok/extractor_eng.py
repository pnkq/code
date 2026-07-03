import unicodedata
from p.dispatcher import Dispatcher

dispatcher = Dispatcher()

with open("corpus_2.txt", "r", encoding="utf-8") as fIn, \
     open("corpus_2_eng.txt", "w", encoding="utf-8") as fOut:

    for line in fIn:
        nfc_line = unicodedata.normalize("NFC", line)
        spans = dispatcher.dispatch(nfc_line)
        words = []
        for span in spans:
            if span.lang == "eng":
                words.append(span.text)
        if words:
            fOut.write(" ".join(words))
            fOut.write("\n")

            