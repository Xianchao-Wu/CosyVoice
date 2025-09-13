#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(utt2wav[utt], 'rb').read() # 这是读取.wav文件里面的内容了
        data_list.append(data)
    wav_list = [utt2wav[utt] for utt in utt_list] # wav file name list
    text_list = [utt2text[utt] for utt in utt_list] # text annotations 
    spk_list = [utt2spk[utt] for utt in utt_list] # speaker
    uttembedding_list = [utt2embedding[utt] for utt in utt_list] # speaker embedding的集合，每个utt一个，utt= case id
    spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list] # 一个speaker对应的平均之后的speaker embedding vector
    speech_token_list = [utt2speech_token.get(utt, []) for utt in utt_list] # case id对应的speech的speech token的序列
    if args.dpo:
        reject_speech_token_list = [utt2reject_speech_token[utt] for utt in utt_list]

    # 保存到parquet,utt2parquet_file,spk2parquet_file
    df = pd.DataFrame()
    df['utt'] = utt_list # case id list
    df['wav'] = wav_list # wav file name list
    df['audio_data'] = data_list # 真实的audio数据
    df['text'] = text_list # all text list
    df['spk'] = spk_list # speaker id list for each case
    df['utt_embedding'] = uttembedding_list # speaker embedding vector list
    df['spk_embedding'] = spkembedding_list # avg speaker embedding vector
    df['speech_token'] = speech_token_list # speech token seq for each case id
    if args.dpo:
        df['reject_speech_token'] = reject_speech_token_list
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split() # '5694_64038_000014_000000 /workspace/asr/CosyVoice/data/tts/openslr/libritts/LibriTTS/dev-clean/5694/64038/5694_64038_000014_000000.wav\n'
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split() # '5694_64038_000014_000000 A MAN IN THE WELL\n'
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split() # '5694_64038_000014_000000 5694\n' file.id speaker.id
            utt2spk[l[0]] = l[1]
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir)) # 5694_64038_000014_000000 -> a list with 192 elements for current voice's speaker embedding vector
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir)) # 5694 -> a list with 192 elements
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir)) # 5694_64038_000014_000000 -> a list of speech tokens
    if args.dpo:
        utt2reject_speech_token = torch.load('{}_reject/utt2speech_token.pt'.format(args.src_dir))
    utts = list(utt2wav.keys()) # case.id alike '2078_142845_000018_000000'

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    pool.close()
    pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
