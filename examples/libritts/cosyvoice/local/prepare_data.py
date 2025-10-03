import argparse
import logging
import glob
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    wavs = list(glob.glob('{}/*/*/*wav'.format(args.src_dir))) # NOTE TODO better change to os.walk to automatically walk through all the .wav files in current dir

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        txt = wav.replace('.wav', '.normalized.txt')
        if not os.path.exists(txt):
            logger.warning('{} do not exsist'.format(txt))
            continue
        with open(txt) as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())
        utt = os.path.basename(wav).replace('.wav', '') # e.g., utt='5694_64038_000014_000000'
        spk = utt.split('_')[0] # e.g., spk=5694, speaker id
        utt2wav[utt] = wav
        utt2text[utt] = content # e.g., 'A MAN IN THE WELL'
        utt2spk[utt] = spk # '5694'
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--ref_model',
                        type=str)
    args = parser.parse_args()
    main()

'''
四个种类的dict:

ipdb> dict utt2wav -> file wav.scp
{'5694_64038_000014_000000': '/workspace/asr/CosyVoice/data/tts/openslr/libritts/LibriTTS/dev-clean/5694/64038/5694_64038_000014_000000.wav'}
->
5694_64038_000014_000000 /workspace/asr/CosyVoice/data/tts/openslr/libritts/LibriTTS/dev-clean/5694/64038/5694_64038_000014_000000.wav




ipdb> dict utt2text -> file text
{'5694_64038_000014_000000': 'A MAN IN THE WELL'}
->
5694_64038_000014_000000 A MAN IN THE WELL




ipdb> utt2spk -> utt2spk
{'5694_64038_000014_000000': '5694'} # 5694=speaker.id
->
5694_64038_000014_000000 5694




ipdb> spk2utt -> spk2utt
{'5694': ['5694_64038_000014_000000']}
->
7976 7976_105575_000008_000000 7976_105575_000018_000010 7976_105575_000008_000007 7976_105575_000004_00...




'''
