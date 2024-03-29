import sys

import torch
from transformers import AutoModel, AutoTokenizer
from text.vietnamese import segment_sentence, text_normalize #, tokenizer

LOCAL_PATH = "./bert/phobert-base-v2" #Using phobert base. Can change path if want to use large version

try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
except:
    print("Can't load Bert from local dir. Download from hub...")
    LOCAL_PATH = "vinai/phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

models = dict()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_bert_feature(
    text,
    word2ph,
    device=device,
    style_text=None,
    style_weight=0.7,
):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModel.from_pretrained(LOCAL_PATH).to(device)
    with torch.no_grad():
        text = text_normalize(text) # Normalize text
        text = segment_sentence(text)
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)
    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
