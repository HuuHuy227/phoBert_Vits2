import torch
import commons
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import utils

from models import SynthesizerTrn
from text.symbols import symbols

def get_net_g(model_path: str, device: str, hps):
    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)


    return net_g


def get_text(text, hps, device, style_text=None, style_weight=0.7):
    style_text = None if style_text == "" else style_text
    norm_text, phone, tone, word2ph = clean_text(text)
    phone, tone = cleaned_text_to_sequence(phone, tone)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(
        norm_text, word2ph, device, style_text, style_weight
    )
    del word2ph

    assert bert.shape[-1] == len(phone), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    return bert,  phone,  tone


def infer(
    text,
    hps,
    net_g,
    device,
    noise_scale = 0.667,
    noise_scale_w = 0.8,
    length_scale = 1,
    skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
):
    bert, phones, tones = get_text(text,hps,device,style_text=style_text,style_weight=style_weight)

    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        bert = bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        bert = bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # emo = emo.to(device).unsqueeze(0)
        del phones
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                tones,
                bert,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del (
            x_tst,
            tones,
            bert,
            x_tst_lengths
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio