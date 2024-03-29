from text.symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text, tones):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return phones, tones

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result

def get_bert(norm_text, word2ph, device, style_text=None, style_weight=0.7):
    from .pho_bert import get_bert_feature as vi_bert
    bert = vi_bert(norm_text, word2ph, device, style_text, style_weight)

    return bert


def check_bert_models():
    import json
    from pathlib import Path

    from config import config
    from .bert_utils import _check_bert

    # if config.mirror.lower() == "openi":
    #     import openi

    #     kwargs = {"token": config.openi_token} if config.openi_token else {}
    #     openi.login(**kwargs)

    with open("./bert/bert_models.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./bert").joinpath(k)
            _check_bert(v["repo_id"], v["files"], local_path)

def check_slm_models():
    import json
    from pathlib import Path

    from config import config
    from .bert_utils import _check_slm

    # if config.mirror.lower() == "openi":
    #     import openi

    #     kwargs = {"token": config.openi_token} if config.openi_token else {}
    #     openi.login(**kwargs)

    with open("./slm/slm_model.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./slm").joinpath(k)
            _check_slm(v["repo_id"], v["files"], local_path)

# def init_openjtalk():
#     import platform

#     if platform.platform() == "Linux":
#         import pyopenjtalk

#         pyopenjtalk.g2p("こんにちは，世界。")


# init_openjtalk()
#check_bert_models()
