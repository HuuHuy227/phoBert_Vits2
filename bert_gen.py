# import torch
# from multiprocessing import Pool
# import commons
# from tqdm import tqdm
# from text import check_bert_models, cleaned_text_to_sequence, get_bert
# import argparse
# import torch.multiprocessing as mp
# import os

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# use_multi_device = False
# audio_path = "wavs"

# def process_line(x):
#     line, add_blank = x
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if use_multi_device:
#         rank = mp.current_process()._identity
#         rank = rank[0] if len(rank) > 0 else 0
#         if torch.cuda.is_available():
#             gpu_id = rank % torch.cuda.device_count()
#             device = torch.device(f"cuda:{gpu_id}")
#         else:
#             device = torch.device("cpu")
#     wav_path, text, phones, tone, word2ph = line.strip().split("|")
#     phone = phones.split(" ")
#     tone = [int(i) for i in tone.split(" ")]
#     word2ph = [int(i) for i in word2ph.split(" ")]
#     word2ph = [i for i in word2ph]
#     phone, tone = cleaned_text_to_sequence(phone, tone)

#     if add_blank:
#         phone = commons.intersperse(phone, 0)
#         tone = commons.intersperse(tone, 0)
#         language = commons.intersperse(language, 0)
#         for i in range(len(word2ph)):
#             word2ph[i] = word2ph[i] * 2
#         word2ph[0] += 1

#     bert_path = wav_path.replace(".wav", ".bert.pt")

#     try:
#         bert = torch.load(os.path.join(audio_path, bert_path))
#         assert bert.shape[0] == 768 # phoBert base change to 2048 if using phoBert Large
#     except Exception:
#         bert = get_bert(text, word2ph, device)
#         assert bert.shape[-1] == len(phone)
#         torch.save(bert, os.path.join(audio_path, bert_path))


# #preprocess_text_config = config.preprocess_text_config

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # parser.add_argument(
#     #     "-c", "--config", type=str, default=config.bert_gen_config.config_path
#     # )
#     parser.add_argument(
#         "-audio_path", "--config", type=str
#     )
#     parser.add_argument(
#         "--num_processes", type=int, default=4
#     )
#     parser.add_argument(
#         "--use_multi_device", type=bool, default=False
#     )
#     parser.add_argument(
#         "--add_blank", type=bool, default=False
#     )
#     parser.add_argument(
#         "--filelists",
#         nargs="+",
#         default=[
#             "filelists/train.txt.cleaned",
#             "filelists/val.txt.cleaned",
#         ],
#     )
#     #args, _ = parser.parse_known_args()
#     # config_path = args.config
#     # hps = utils.get_hparams_from_file(config_path)
#     args = parser.parse_args()
#     check_bert_models()
#     lines = []

#     for filelist in args.filelists:
#         with open(filelist, encoding="utf-8") as f:
#             lines.extend(f.readlines())

#     # with open(hps.data.validation_files, encoding="utf-8") as f:
#     #     lines.extend(f.readlines())
#     add_blank = [args.add_blank] * len(lines)

#     if len(lines) != 0:
#         num_processes = args.num_processes
#         with Pool(processes=num_processes) as pool:
#             for _ in tqdm(
#                 pool.imap_unordered(process_line, zip(lines, add_blank)),
#                 total=len(lines),
#             ):
#                 pass  

#     print(f"bert created!,{len(lines)} bert.pt has created!")


import torch
from multiprocessing import Pool
import commons
from tqdm import tqdm
from text import check_bert_models, cleaned_text_to_sequence, get_bert
import argparse
import torch.multiprocessing as mp
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_multi_device = False
audio_path = "wavs"

def process_line(x):
    line, add_blank = x
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone = cleaned_text_to_sequence(phone, tone)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".wav", ".bert.pt")

    try:
        bert = torch.load(os.path.join(audio_path, bert_path))
        assert bert.shape[0] == 768 # phoBert base change to 2048 if using phoBert Large
    except Exception:
        bert = get_bert(text, word2ph, device)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, os.path.join(audio_path, bert_path))


#preprocess_text_config = config.preprocess_text_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c", "--config", type=str, default=config.bert_gen_config.config_path
    # )
    parser.add_argument(
        "--audio_path", "--config", type=str
    )
    parser.add_argument(
        "--num_processes", type=int, default=4
    )
    parser.add_argument(
        "--use_multi_device", type=bool, default=False
    )
    parser.add_argument(
        "--add_blank", type=bool, default=False
    )
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "filelists/train.txt.cleaned",
            "filelists/val.txt.cleaned",
        ],
    )
    #args, _ = parser.parse_known_args()
    # config_path = args.config
    # hps = utils.get_hparams_from_file(config_path)
    args = parser.parse_args()
    check_bert_models()
    lines = []

    for filelist in args.filelists:
        with open(filelist, encoding="utf-8") as f:
            lines.extend(f.readlines())

    # with open(hps.data.validation_files, encoding="utf-8") as f:
    #     lines.extend(f.readlines())
    add_blank = [args.add_blank] * len(lines)

    # count = 0
    # for line in tqdm(lines):
    #     try:
    #         process_line(line)
    #     except:
    #         count += 1
    #         print("Can't create Bert. Skipping!!")
    #         continue

    if len(lines) != 0:
        num_processes = args.num_processes
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_line, zip(lines, add_blank)), #, args.use_multi_device, args.audio_path)),
                total=len(lines),
            ):
                pass  

    print(f"bert created!,{len(lines)} bert.pt has created!")
