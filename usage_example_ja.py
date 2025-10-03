import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# NOTE inference_zero_shot
#for i, j in enumerate(cosyvoice.inference_zero_shot('遠く離れた友人から誕生日プレゼントをもらい、思いがけない驚きと深い祝福で心が甘い喜びで満たされ、笑顔が花のように咲きました。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#    import ipdb; ipdb.set_trace()
#    torchaudio.save('1_ja_zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

#for i, j in enumerate(cosyvoice.inference_zero_shot('とおく はなれ た ゆうじん から たんじょう び ぷれぜんと を もらい 、 おもいがけない おどろき と ふかい しゅくふく で こころ が あまい よろこび で みたさ れ 、 えがお が はな の よう に さき まし た 。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_zero_shot('とおく はなれ た ゆうじん から たんじょう び ぷれぜんと を もらい 、 おもいがけない おどろき と ふかい しゅくふく で こころ が あまい よろこび で みたさ れ 、 えがお が はな の よう に さき まし た 。'.replace(' ', ''), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
for i, j in enumerate(cosyvoice.inference_zero_shot('r i N g o o s a N k o t o o r e N j i o n i j u u g o k i r o g u r a m u k a i m a sh I t a'.replace(' ', ''), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    #import ipdb; ipdb.set_trace()
    torchaudio.save('1_ja_zimuin_zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

examples = [
    "私は2025年10月3日に東京へ行きます。",
    "会議は午後3時15分から始まります。",
    "りんごを3個とオレンジを25kg買いました。",
    "今年の利益率は12.5％で、売上は3億5000万円でした。",
    "平成31年4月30日に退位されました。",
    "私はコンピュータとインターネットをよく使います。",
    "明日のMeetingはZoomで13:00からです。",
    "次の候補は（A）田中さん、（B）佐藤さんです。",
    "やったーーー！！！今日はめっちゃ楽しい〜〜〜",
    "今日は最高😊✨ 100点満点！",
]
aid=0
for example in examples:
    aid+=1
    for i, j in enumerate(cosyvoice.inference_zero_shot(example.replace(' ', ''), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        #import ipdb; ipdb.set_trace()
        torchaudio.save('1_ja_hanzi_zero_shot_{}_id{}.wav'.format(i, aid), j['tts_speech'], cosyvoice.sample_rate)


'''
  → 平假名/片假名2: わたしわにせんにじゅーごねんじゅーがつみっかにとーきょーえいきます。
  → 平假名/片假名2: かいぎわごごさんじじゅーごふんからはじまります。
  → 平假名/片假名2: りんごをさんことおれんじをにじゅーごきろぐらむかいました。
  → 平假名/片假名2: ことしのりえきりつわじゅーにーてんごぱーせんとで、うりあげわさんおくごせんまんえんでした。
  → 平假名/片假名2: へーせーさんじゅーいちねんしがつさんじゅーにちにたいいされました。
  → 平假名/片假名2: わたしわこんぴゅーたといんたーねっとをよくつかいます。
  → 平假名/片假名2: あしたのみーてぃんぐわずーむでじゅーさん：ぜろぜろからです。
  → 平假名/片假名2: つぎのこーほわ（Ａ）たなかさん、（Ｂ）さとーさんです。
  → 平假名/片假名2: やったーーー！！！きょーわめっちゃたのしい〜〜〜
  → 平假名/片假名2: きょーわさいこー😊✨　ひゃくてんまんてん！
'''
examples = [
    'わたしわにせんにじゅーごねんじゅーがつみっかにとーきょーえいきます。',
    'かいぎわごごさんじじゅーごふんからはじまります。',
    'りんごをさんことおれんじをにじゅーごきろぐらむかいました。',
    'ことしのりえきりつわじゅーにーてんごぱーせんとで、うりあげわさんおくごせんまんえんでした。',
    'へーせーさんじゅーいちねんしがつさんじゅーにちにたいいされました。',
    'わたしわこんぴゅーたといんたーねっとをよくつかいます。',
    'あしたのみーてぃんぐわずーむでじゅーさん：ぜろぜろからです。',
    'つぎのこーほわ（Ａ）たなかさん、（Ｂ）さとーさんです。',
    'やったーーー！！！きょーわめっちゃたのしい〜〜〜',
    'きょーわさいこー😊✨　ひゃくてんまんてん！',
]
aid=0
for example in examples:
    aid+=1
    for i, j in enumerate(cosyvoice.inference_zero_shot(example.replace(' ', ''), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        #import ipdb; ipdb.set_trace()
        torchaudio.save('1_ja_hiragana_zero_shot_{}_id{}.wav'.format(i, aid), j['tts_speech'], cosyvoice.sample_rate)



sys.exit(0)

# NOTE add_zero_shot_spk
# save zero_shot spk for future usage
assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True # TODO 重要的是，这里的'my_zero_shot_spk'对应的speaker embedding vector信息，是之前的reference voice对应的speaker embedding vector拿到的.

# NOTE inference_zero_shot
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)): # TODO 需要注意的是，这里使用的是已有的speaker vector，"my_zero_shot_spk"。
    import ipdb; ipdb.set_trace()
    torchaudio.save('2_zero_shot_my_zero_shot_spk{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
cosyvoice.save_spkinfo()

# NOTE inference_cross_lingual
# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    import ipdb; ipdb.set_trace()
    torchaudio.save('3_fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# NOTE inference_instruct2
# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    import ipdb; ipdb.set_trace()
    torchaudio.save('4_instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

# NOTE inference_zero_shot
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    import ipdb; ipdb.set_trace()
    torchaudio.save('5_zero_shot_xiwang_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

