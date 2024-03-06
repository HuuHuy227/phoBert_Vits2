import re
import os

import py_vncorenlp
from viphoneme import vi2IPA_split_seg_word

from text import symbols
from vinorm import TTSnorm
from text.symbols import punctuation

from transformers import AutoTokenizer

LOCAL_PATH = "./bert/phobert-base-v2" #Using phobert base. Can change path if want to use large version
path = os.path.join(os.getcwd(),"py_vncorenlp") #Get current path

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=path)

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
    # if text[-1] == ".":
    #     text = text[:-1] # Remove last end point because it causes conflict with viphoneme
    # text = replace_all(text, dict_map)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    
    text = TTSnorm(text)
    text = text.strip() #Text norm from vinorm
    return text

def segment_sentence(text):
    if text[-1] != '.':
        text += '.' # Need to add end point (.) if it doesn't exist.
    seg_text = rdrsegmenter.word_segment(text) 
    return seg_text

def refine_ph(phn):
    """
        Refine phone
        Input: phone
        Output: Calculate phone and tone
    """
    tone = 0
    if re.search(r"\d", phn[:-1]):
        tone = int(phn[-2])
        phn = phn[:-2]
    elif not re.search(r"\d", phn[:-1]) and len(phn) > 3:
        tone = 1
    else:
        tone = 0
    return phn, tone

def refine_tok(phonem, tokens):
    """
        Refine tokenizer between phoTokenizer and word segment
        Input: phonemizer, tokens of phoTokenizer
        Output: Refine phonem with length equal to length tokens
        O(n^2) ReThinking to optimize 
    """
    i = 0
    j = 0
    refine_tok = []
    n = len(tokens)
    while i < (n - 1): 
        if "@@" in tokens[i]:
            for k in range(i, n):
                if "@@" not in tokens[k]: # Break the loop
                    refine_tok.append(phonem[j])
                    i = k + 1
                    j += 1
                    break
                if "_" in tokens[i]:
                    eles = phonem[j].split("_")
                    phonem[j] = "_".join(eles[1:])
                    refine_tok.append(eles[0])
                elif tokens[k] == "<unk>":
                    refine_tok.append("UNK")
                else:
                    ele = phonem[j].split("/")[1:-1]
                    #print(ele)
                    if ele[0].isnumeric(): #Check if is number
                        refine_tok.append("UNK")
                        phonem[j] = "/".join(ele[1:]).replace("_","")
                    else:
                        refine_tok.append("/" + ele[0] + "/" +"1")
                        phonem[j] = "/" + "/".join(ele[1:]) # Remove
                    i += 1
        elif tokens[i] == '<unk>': # Handle TH unk token
            refine_tok.append("UNK")
            i += 1
        else:
            refine_tok.append(phonem[j])
            j += 1
            i += 1    
        # print(i,j)
        #print(refine_tok)
        #print(phonem)
    if '/./' != refine_tok[-1]:
        refine_tok.append('/./')
    return refine_tok

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
    word_seg = segment_sentence(text)

    phonemes = vi2IPA_split_seg_word(word_seg,delimit="/")
    #print(phonemes)
    phonemes = phonemes.split()
    phonemes = phonemes[:-1] #Phonemes contain 2 point /./ /./ So we must to remove last point (/./)
    
    input_ids = tokenizer.encode(word_seg[0])
    toks = [tokenizer._convert_id_to_token(ids) for ids in input_ids[1:-1]]
    #print(toks)
    #print(len(toks), len(phonemes))
    if len(toks) != len(phonemes): #Handle conflict between phoTokenizer and word segments
        words = refine_tok(phonemes, toks)
    else:
        words = phonemes
    #print(words)
    assert len(words) == len(toks)
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


