import torch
#from multiprocessing import Pool
import commons
from tqdm import tqdm
from text import check_bert_models, cleaned_text_to_sequence, get_bert
import argparse
#import torch.multiprocessing as mp
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_line(line, audio_path, out_path = None):
    wav_path, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone = cleaned_text_to_sequence(phone, tone)

    bert_path = wav_path.replace(".wav", ".bert.pt")

    # bert = get_bert(text, word2ph, device)
    # assert bert.shape[-1] == len(phone)
    # if not out_path: 
    #   torch.save(bert, os.path.join(audio_path, bert_path))
    # else:
    #   torch.save(bert, os.path.join(out_path, bert_path))

    try:
        if not out_path:
            bert = torch.load(os.path.join(audio_path, bert_path))
        else:
            bert = torch.load(os.path.join(out_path, bert_path))
        assert bert.shape[0] == 1536 # phoBert base change to 2048 if using phoBert Large
    except Exception:
        bert = get_bert(text, word2ph, device)
        assert bert.shape[-1] == len(phone)
        if not out_path: 
            torch.save(bert, os.path.join(audio_path, bert_path))
        else:
            torch.save(bert, os.path.join(out_path, bert_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c", "--config", type=str, default=config.bert_gen_config.config_path
    # )
    parser.add_argument(
        "--audio_path", type=str
    )
    parser.add_argument(
        "--out_path", type=str, default=None
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

    args = parser.parse_args()
    check_bert_models()
    lines = []

    for filelist in args.filelists:
        with open(filelist, encoding="utf-8") as f:
            lines.extend(f.readlines())

    count = 0
    for line in tqdm(lines, total=len(lines), position=0, leave=True):
        #process_line(line, args.audio_path, args.out_path)
        try:
            process_line(line, args.audio_path, args.out_path)
        except:
            count += 1
            print("Can't create Bert. Skipping!!")
            continue

    print(f"bert created!,{len(lines) - count} bert.pt has created!")


# import torch
# from multiprocessing import Pool
# import commons
# from tqdm import tqdm
# from text import check_bert_models, cleaned_text_to_sequence, get_bert
# import argparse
# import torch.multiprocessing as mp
# import os

# # def process_line(line, audio_path, out_path = None):
# #     # line, add_blank = x
# #     # if use_multi_device:
# #     #     rank = mp.current_process()._identity
# #     #     rank = rank[0] if len(rank) > 0 else 0
# #     #     if torch.cuda.is_available():
# #     #         gpu_id = rank % torch.cuda.device_count()
# #     #         device = torch.device(f"cuda:{gpu_id}")
# #     #     else:
# #     #         device = torch.device("cpu")
# #     wav_path, text, phones, tone, word2ph = line.strip().split("|")
# #     phone = phones.split(" ")
# #     tone = [int(i) for i in tone.split(" ")]
# #     word2ph = [int(i) for i in word2ph.split(" ")]
# #     word2ph = [i for i in word2ph]
# #     phone, tone = cleaned_text_to_sequence(phone, tone)

# #     # if add_blank:
# #     #     phone = commons.intersperse(phone, 0)
# #     #     tone = commons.intersperse(tone, 0)
# #     #     language = commons.intersperse(language, 0)
# #     #     for i in range(len(word2ph)):
# #     #         word2ph[i] = word2ph[i] * 2
# #     #     word2ph[0] += 1

# #     bert_path = wav_path.replace(".wav", ".bert.pt")

# #     bert = get_bert(text, word2ph, device)
# #     assert bert.shape[-1] == len(phone)
# #     if not out_path: 
# #       torch.save(bert, os.path.join(audio_path, bert_path))
# #     else:
# #       torch.save(bert, os.path.join(out_path, bert_path))

# #     # try:
# #     #     bert = torch.load(os.path.join(audio_path, bert_path))
# #     #     assert bert.shape[0] == 768 # phoBert base change to 2048 if using phoBert Large
# #     # except Exception:
# #     #     bert = get_bert(text, word2ph, device)
# #     #     assert bert.shape[-1] == len(phone)
# #     #     if not out_path: 
# #     #         torch.save(bert, os.path.join(audio_path, bert_path))
# #     #     else:
# #     #         torch.save(bert, os.path.join(out_path, bert_path))

# def process_line(x):
#     line, use_multi_device, out_path, device = x
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

#     bert_path = wav_path.replace(".wav", ".bert.pt")

#     try:
#         bert = torch.load(bert_path)
#         assert bert.shape[0] == 1536
#     except Exception:
#         bert = get_bert(text, word2ph, device)
#         assert bert.shape[-1] == len(phone)
#         torch.save(bert, os.path.join(out_path, bert_path))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # parser.add_argument(
#     #     "-c", "--config", type=str, default=config.bert_gen_config.config_path
#     # )
#     parser.add_argument(
#         "--num_processes", type=int, default=4
#     )
#     parser.add_argument(
#         "--out_path", type=str, default=None
#     )
#     parser.add_argument(
#         "--use_multi_device", type=bool, default=False
#     )
#     parser.add_argument(
#         "--filelists",
#         nargs="+",
#         default=[
#             "filelists/train.txt.cleaned",
#             "filelists/val.txt.cleaned",
#         ],
#     )

#     args = parser.parse_args()
#     check_bert_models()
#     lines = []
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     for filelist in args.filelists:
#         with open(filelist, encoding="utf-8") as f:
#             lines.extend(f.readlines())

#     devices = [device] * len(lines)
#     use_multi_devices = [args.use_multi_device] * len(lines)
#     out_paths = [args.out_path] * len(lines)

#     # for line in tqdm(lines, total=len(lines), position=0, leave=True):
#     #     process_line(line, args.audio_path, args.out_path)
#     #     # try:
#     #     #     process_line(line, args.audio_path, args.out_path)
#     #     # except:
#     #     #     count += 1
#     #     #     print("Can't create Bert. Skipping!!")
#     #     #     continue
    
#     if len(lines) != 0:
#         with Pool(processes=args.num_processes) as pool:
#             for _ in tqdm(
#                 pool.imap_unordered(process_line, zip(lines, use_multi_devices, out_paths, devices)),
#                 total=len(lines),
#             ):
#                 pass  
#     print(f"bert created!,{len(lines)} bert.pt has created!")
