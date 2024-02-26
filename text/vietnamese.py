import re

from viphoneme import vi2IPA_split

from text import symbols
from text.symbols import punctuation

def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",", 
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }

def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text

def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    return replaced_text

_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_number_re = re.compile(r"[0-9]+")

def _expand_decimal_point(m):
    return m.group(1).replace(".", " chấm ")

def normalize_numbers(text):
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    return text

def text_normalize(text):
    #text = normalize_numbers(text)
    if text[-1] == ".":
        text = text[:-1] # Remove last end point because it causes conflict with viphoneme
    text = replace_all(text, dict_map)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text

def refine_ph(phn):
    tone = 0
    if re.search(r"\d", phn[:-1]):
        tone = int(phn[-2])
        phn = phn[:-2]
    elif not re.search(r"\d", phn[:-1]) and len(phn) > 3:
        tone = 1
    else:
        tone = 0
    return phn, tone

def cal_ph(word):
    ph, tn = refine_ph(word)
    tmp_ph = ph.split("/")[1:-1]
    # Create 1 tone for all phoneome if phono in punction is 0
    tmp_tone = [0 if p in punctuation else 1 for p in tmp_ph] # In case English text, all will have tone [1,1,...,1]
    if len(tmp_tone) > 1:
        tmp_tone[1] = tn # Tone for vietnamese usually in 2nd phoneme

    return tmp_ph, tmp_tone

def g2p(text):
    phones = []
    tones = []
    word2ph = []

    text = text.replace('\s+',' ').lower()
    words = vi2IPA_split(text,delimit="/").split()

    for word in words:
        if "_" in word: # handle TH tu ghep vd: vi_tri nghien_cuu_vien, ...
            ph_count = 0
            w_split = word.split("_")
            for w in w_split:
                tmp_ph, tmp_tone = cal_ph(w)
                    
                tones.extend(tmp_tone)
                phones.extend(tmp_ph)
                ph_count += len(tmp_ph)
            word2ph.append(ph_count)

        else: # TH tu don vd: thuong, phong , nha, xe, ...
            tmp_ph, tmp_tone = cal_ph(word)
                
            tones.extend(tmp_tone)
            phones.extend(tmp_ph)
            word2ph.append(len(tmp_ph))

    phones = ["_"] + phones + ["_"] #Add _ to head and tail of phones to suitable for emb tokenizer of phoBert (<bos> .. <eos>)
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    return phones, tones, word2ph

def get_bert_feature(text, word2ph):
    from text import pho_bert

    return pho_bert.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    phones, tones, word2ph = g2p("Xin chào, tên của tôi là Huy")
    emb = get_bert_feature("Xin chào, tên của tôi là Huy.", word2ph)
    print(phones)
    print(tones)
    print(word2ph)
    print(emb.size())


