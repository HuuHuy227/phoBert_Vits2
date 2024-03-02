# import os
# import re
# # from g2p_en import G2p
# # from transformers import DebertaV2Tokenizer


# from viphoneme import vi2IPA_split
# #print(vi2IPA_split("Bây giờ là 10h30. Tôi học có 5.5 điểm. Manchester nên tôi buồn vãi chấm.", delimit = "/"))

# from text import symbols
# from text.symbols import punctuation

# current_file_path = os.path.dirname(__file__)
# CMU_DICT_PATH = os.path.join(current_file_path, "text/cmudict.rep")
# CACHE_PATH = os.path.join(current_file_path, "text/cmudict_cache.pickle")
# # _g2p = G2p()
# # LOCAL_PATH = "./bert/deberta-v3-large"
# # tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

# def post_replace_ph(ph):
#     rep_map = {
#         "：": ",",
#         "；": ",",
#         "，": ",", 
#         "。": ".",
#         "！": "!",
#         "？": "?",
#         "\n": ".",
#         "·": ",",
#         "、": ",",
#         "…": "...",
#         "···": "...",
#         "・・・": "...",
#         "v": "V",
#     }
#     if ph in rep_map.keys():
#         ph = rep_map[ph]
#     if ph in symbols:
#         return ph
#     if ph not in symbols:
#         ph = "UNK"
#     return ph

# rep_map = {
#     "：": ",",
#     "；": ",",
#     "，": ",",
#     "。": ".",
#     "！": "!",
#     "？": "?",
#     "\n": ".",
#     "．": ".",
#     "…": "...",
#     "···": "...",
#     "・・・": "...",
#     "·": ",",
#     "・": ",",
#     "、": ",",
#     "$": ".",
#     "“": "'",
#     "”": "'",
#     '"': "'",
#     "‘": "'",
#     "’": "'",
#     "（": "'",
#     "）": "'",
#     "(": "'",
#     ")": "'",
#     "《": "'",
#     "》": "'",
#     "【": "'",
#     "】": "'",
#     "[": "'",
#     "]": "'",
#     "—": "-",
#     "−": "-",
#     "～": "-",
#     "~": "-",
#     "「": "'",
#     "」": "'",
# }


# def replace_punctuation(text):
#     pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

#     replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

#     # replaced_text = re.sub(
#     #     r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
#     #     + "".join(punctuation)
#     #     + r"]+",
#     #     "",
#     #     replaced_text,
#     # )

#     return replaced_text

# # def refine_ph(phn):
# #     tone = 0
# #     if re.search(r"\d$", phn):
# #         tone = int(phn[-1]) + 1
# #         phn = phn[:-1]
# #     else:
# #         tone = 3
# #     return phn.lower(), tone


# # def refine_syllables(syllables):
# #     tones = []
# #     phonemes = []
# #     for phn_list in syllables:
# #         for i in range(len(phn_list)):
# #             phn = phn_list[i]
# #             phn, tone = refine_ph(phn)
# #             phonemes.append(phn)
# #             tones.append(tone)
# #     return phonemes, tones

# _decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
# # _dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
# # _ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
# _number_re = re.compile(r"[0-9]+")

# # # List of (regular expression, replacement) pairs for abbreviations: Need to do for Vietnamese later
# # _abbreviations = [
# #     (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
# #     for x in [
# #         ("mrs", "misess"),
# #         ("mr", "mister"),
# #         ("dr", "doctor"),
# #         ("st", "saint"),
# #         ("co", "company"),
# #         ("jr", "junior"),
# #         ("maj", "major"),
# #         ("gen", "general"),
# #         ("drs", "doctors"),
# #         ("rev", "reverend"),
# #         ("lt", "lieutenant"),
# #         ("hon", "honorable"),
# #         ("sgt", "sergeant"),
# #         ("capt", "captain"),
# #         ("esq", "esquire"),
# #         ("ltd", "limited"),
# #         ("col", "colonel"),
# #         ("ft", "fort"),
# #     ]
# # ]


# def _expand_decimal_point(m):
#     return m.group(1).replace(".", " chấm ")


# def normalize_numbers(text):
#     text = re.sub(_decimal_number_re, _expand_decimal_point, text)
#     return text

# def text_normalize(text):
#     text = normalize_numbers(text)
#     text = replace_punctuation(text)
#     text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
#     return text

# # def distribute_phone(n_phone, n_word):
# #     phones_per_word = [0] * n_word
# #     for task in range(n_phone):
# #         min_tasks = min(phones_per_word)
# #         min_index = phones_per_word.index(min_tasks)
# #         phones_per_word[min_index] += 1
# #     return phones_per_word


# # def sep_text(text):
# #     words = re.split(r"([,;.\?\!\s+])", text)
# #     words = [word for word in words if word.strip() != ""]
# #     return words

# def refine_ph(phn):
#     tone = 0
#     if re.search(r"\d", phn[:-1]):
#         tone = int(phn[-2])
#         phn = phn[:-2]
#     elif not re.search(r"\d", phn[:-1]) and len(phn) > 3:
#         tone = 1
#     else:
#         tone = 0
#     return phn, tone

# #print(refine_ph("/./"))


# def g2p(text):
#     phones = []
#     tones = []
#     word2ph = []
#     # words = sep_text(text)
#     # tokens = [tokenizer.tokenize(i) for i in words]
#     # words = text_to_words(text)
#     # print(words)

#     text = text.replace('\s+',' ').lower()
#     words = vi2IPA_split(text,delimit="/").split()
#     #print(words)
#     for word in words:
#         ph, tn = refine_ph(word)
#         tmp_ph = ph.split("/")[1:-1]
#         # Create 1 tone for all phoneome if phono in punction is 0
#         tmp_tone = [0 if p in punctuation else 1 for p in tmp_ph] # In case English text, all will have tone [1,1,...,1]
#         if len(tmp_tone) > 1:
#             tmp_tone[1] = tn # Tone for vietnamese usually in 2nd phoneme
        
#         # Create last point if sentence don't have ...
            
#         tones.extend(tmp_tone)
#         phones.extend(tmp_ph)
#         word2ph.append(len(tmp_ph))

#     phones = ["_"] + phones + ["_"] #Add _ to head and tail of phones to suitable for emb tokenizer of phoBert
#     tones = [0] + tones + [0]
#     word2ph = [1] + word2ph + [1]
#     assert len(phones) == len(tones), text
#     assert len(phones) == sum(word2ph), text

#     return phones, tones, word2ph

# def get_bert_feature(text, word2ph):
#     from text import pho_bert

#     return pho_bert.get_bert_feature(text, word2ph)


# if __name__ == "__main__":
#     # print(get_dict())
#     # print(eng_word_to_phoneme("hello"))
#     phones, tones, word2ph = g2p("Xin chào, tên của tôi là Huy")
#     #emb = get_bert_feature("Xin chào, tên của tôi là Huy.", word2ph) #(1024, len(phones))
#     # print(phones)
#     # print(tones)
#     # print(word2ph)
#     # print(emb.size())
#     # all_phones = set()
#     # for k, syllables in eng_dict.items():
#     #     for group in syllables:
#     #         for ph in group:
#     #             all_phones.add(ph)
#     # print(all_phones)

#     import torch
#     from models import SynthesizerTrn

#     net_g = SynthesizerTrn(
#         n_vocab=256,
#         spec_channels=80, # <--- vits2 parameter (changed from 513 to 80)
#         segment_size=8192,
#         inter_channels=192,
#         hidden_channels=192,
#         filter_channels=768,
#         n_heads=2,
#         n_layers=6,
#         kernel_size=3,
#         p_dropout=0.1,
#         resblock="1", 
#         resblock_kernel_sizes=[3, 7, 11],
#         resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
#         upsample_rates=[8, 8, 2, 2],
#         upsample_initial_channel=512,
#         upsample_kernel_sizes=[16, 16, 4, 4],
#         n_speakers=0,
#         gin_channels=256,
#         use_sdp=True, 
#         use_transformer_flows=True, # <--- vits2 parameter
#         # (choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual")
#         transformer_flow_type="fft", # <--- vits2 parameter 
#         use_spk_conditioned_encoder=True, # <--- vits2 parameter
#         use_noise_scaled_mas=True, # <--- vits2 parameter
#         use_duration_discriminator=True, # <--- vits2 parameter
#     )

#     # x = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1,1,1,1,1,1,1,1,1,1,1,1]).unsqueeze(0) # token ids
#     # x_lengths = torch.LongTensor([22, 1]) # token lengths
#     # y = torch.randn(1, 80, 100) # mel spectrograms
#     # y_lengths = torch.Tensor([100, 80]) # mel spectrogram lengths


#     x = torch.LongTensor([[1, 2, 3],[4, 5, 6]]) # token ids [2,34]
#     x_lengths = torch.LongTensor([2, 3]) # token lengths
#     y = torch.randn(2, 80, 100) # mel spectrograms
#     y_lengths = torch.Tensor([100, 80]) # mel spectrogram lengths

#     tones = torch.LongTensor([[1,1,3],[1,3,4]])
    
#     bert = torch.randn(2, 1024,3)
#     # print(bert.size())
#     res = net_g(
#         x=x,
#         x_lengths=x_lengths,
#         y=y,
#         y_lengths=y_lengths,
#         tone = tones,
#         bert = bert
#     )
#     #print(res)
#     # calculate loss and backpropagate

from text.vietnamese import g2p
# from text import vietnamese
# from underthesea import word_tokenize
# from text import vietnamese, cleaned_text_to_sequence
from text.cleaner import clean_text
from text import check_bert_models, cleaned_text_to_sequence, get_bert
# from text.pho_bert import get_bert_feature
# text = "lạc long quân lấy âu cơ sinh ra một bọc trăm trứng nở ra một trăm người con là tổ tiên của người bách việt"
text = "tiên nhân phật tổ tu la a tu la bồ tát ma vương quỷ đế"
# text = "dân gian thường lấy thịt và xương trăn để nấu cao bồi bổ sức khỏe tráng kiện gân"
# text = "hạt nhân năm ngoái và một loạt thử nghiệm tên lửa đạn đạo"
phones, tones, word2ph = g2p(text)

# check_bert_models()
# w2ph = [1, 3, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 5, 2, 3, 3, 3, 1, 1]
# bert = get_bert_feature(text, w2ph, device="cuda")
# print(bert.size())
# print(word_tokenize(text,format="text"))
# print(nortext)
# print(phones)
# print(tones)
# print(word2ph)



# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder

# import os
# import py_vncorenlp
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.dirname(os.path.abspath(__file__)) + '\py_vncorenlp')

# from viphoneme import vi2IPA_split_seg_word, vi2IPA_split
# text = "không kê giường sát cửa sổ để tránh những ảnh hưởng không tốt về sức khỏe"
# print(vi2IPA_split(text, delimit="/"))
# text = rdrsegmenter.word_segment(text)
# print(text)
# print(vi2IPA_split_seg_word(text, delimit="/"))

# import os
# print(os.environ.get('JAVA_HOME'))

# from viphoneme import vi2IPA_split
# text = "ca sĩ thái tèo linh chia sẻ con chị không mắc chứng tự kỷ"
# text = vietnamese.text_normalize(text)
# #print(text)
# phonem = vietnamese.vi2IPA_split(text, delimit="/").split()
# print(phonem)

# from transformers import AutoTokenizer
# from underthesea import word_tokenize

# LOCAL_PATH = "bert/phobert-base-v2" #Using phobert base. Can change path if want to use large version

# text = word_tokenize(text, format="text")
# tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
# input_ids = tokenizer.encode(text)
# # input_ids
# #print(text)
# id2tok = [tokenizer._convert_id_to_token(ids) for ids in input_ids[1:-1]]
# print(len(id2tok))
#print(refine_tok(phonem, id2tok))




# print(vietnamese.text_normalize("Xin chào, tên của tôi là Huy 5.5333"))

# import torch
# from models import SynthesizerTrn

# net_g = SynthesizerTrn(
#         n_vocab=256,
#         spec_channels=80, # <--- vits2 parameter (changed from 513 to 80)
#         segment_size=8192,
#         inter_channels=192,
#         hidden_channels=192,
#         filter_channels=768,
#         n_heads=2,
#         n_layers=6,
#         kernel_size=3,
#         p_dropout=0.1,
#         resblock="1", 
#         resblock_kernel_sizes=[3, 7, 11],
#         resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
#         upsample_rates=[8, 8, 2, 2],
#         upsample_initial_channel=512,
#         upsample_kernel_sizes=[16, 16, 4, 4],
#         n_speakers=0,
#         gin_channels=0,
#         use_sdp=True, 
#         # use_transformer_flows=True, # <--- vits2 parameter
#         # # (choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual")
#         # transformer_flow_type="fft", # <--- vits2 parameter 
#         use_spk_conditioned_encoder=False, # <--- vits2 parameter
#         use_noise_scaled_mas=True, # <--- vits2 parameter
#         use_duration_discriminator=True, # <--- vits2 parameter
#     )

# x = torch.LongTensor([[1, 2, 3],[4, 5, 6]]) # token ids [2,34]
# x_lengths = torch.LongTensor([2, 3]) # token lengths
# y = torch.randn(2, 80, 100) # mel spectrograms
# y_lengths = torch.Tensor([100, 80]) # mel spectrogram lengths

# tones = torch.LongTensor([[1,1,3],[1,3,4]])
    
# bert = torch.randn(2, 768,3)
#     # print(bert.size())
# res = net_g(
#         x=x,
#         x_lengths=x_lengths,
#         y=y,
#         y_lengths=y_lengths,
#         tone = tones,
#         bert = bert
#     )