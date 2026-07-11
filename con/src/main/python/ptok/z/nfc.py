import unicodedata

text = "Trong ngành công nghiệp bảo hiểm, công nghệ đang được ứng dụng ở mức nào? Chiến lược chuyển đổi số của các doanh nghiệp bảo hiểm hiện nay đã và đang mang lại những thay đổi đáng kể ra sao? "

# 1. Standardize the text into both strict formats
nfc_text = unicodedata.normalize('NFC', text)
nfd_text = unicodedata.normalize('NFD', text)

# 2. Compare the original text to the standardized formats
is_nfc = (text == nfc_text)
is_nfd = (text == nfd_text)

print(f"Is the text fully Composed (NFC)?   --> {is_nfc}")
print(f"Is the text fully Decomposed (NFD)? --> {is_nfd}")
print(f"Total character length in NFC: {len(nfc_text)}")
print(f"Total character length in NFD: {len(nfd_text)}")


# Extracting the exact word "đổi" from your text
word = "đổi"

print(f"Word: '{word}'")
print(f"Total character length: {len(word)}")
print("-" * 30)

# Print the exact hex code point for each character
for char in word:
    hex_code = f"U+{ord(char):04X}"
    name = unicodedata.name(char)
    print(f"Character: {char}  |  Code: {hex_code}  |  ({name})")