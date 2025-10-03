# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        #import ipdb; ipdb.set_trace()
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder # TODO
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder # TODO
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]: # NOTE for model training
        #import ipdb; ipdb.set_trace() # NOTE forward of class MaskedDiffWithXvec
        token = batch['speech_token'].to(device) # [20, 56], pad.id=0; 语音-> speech tokenizer -> speech_token
        token_len = batch['speech_token_len'].to(device) # [20]
        feat = batch['speech_feat'].to(device) # [20, 95, 80], pad.id=0; 语音 -> 梅尔谱 NOTE 这个就是我们真正的目标数据！flow matching预测出来的也是梅尔谱
        feat_len = batch['speech_feat_len'].to(device) # [20], tensor([82, 95, 94, 93, 93 ...
        embedding = batch['embedding'].to(device) # [20, 192], speaker embedding vectors

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding) # [20, 192] -> Linear(in_features=192, out_features=80, bias=True) -> [20, 80]

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device) # [20, 56, 1], 0 as pad.id
        token = self.input_embedding(torch.clamp(token, min=0)) * mask # speech token (20, 56) -> emb(4096, 512) -> (20, 56, 512) 

        # speech token's encode
        h, h_lengths = self.encoder(token, token_len) # encoder for speech token, h.shape=[20, 56, 512], h_lengths.shape=[20, 1, 56] with True for with token and False for padding
        h = self.encoder_proj(h) # [20, 56, 512] -> Linear(in_features=512, out_features=80, bias=True) -> [20, 56, 80]
        h, h_lengths = self.length_regulator(h, feat_len) # multi-layer conv1d # NOTE 这个很重要，是基于梅尔谱的长度，来对speech token的长度进行regulator了。h从[20, 56, 80] -> length_regulator -> [20, 95, 80]; h_lengths=[20] with tensor([82, 95, 94, 93, 93, 90, 89, 89, 83, 83, 24, 81, 80, 74, 67, 64, 62, 60,...]

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device) # conds.shape=[20, 95, 80], all 0
        for i, j in enumerate(feat_len):
            if random.random() < 0.5: # NOTE TODO why 0.5?
                continue
            index = random.randint(0, int(0.3 * j)) # NOTE 
            conds[i, :index] = feat[i, :index] # 前面(最多)30%的来自音频的梅尔谱，给conds, 这也符合论文中说的"a masked version of X_1 by setting continuous frames to zeros from a random start point to the end!" 即: conds[i, index:]的内容都是0了, 合理
        conds = conds.transpose(1, 2) # [20, 95, 80] -> [20, 80, 95]

        mask = (~make_pad_mask(feat_len)).to(h) # mask.shape=[20, 95] 这个就是纯mask feat的
        # NOTE this is unnecessary, feat/h already same shape
        #import ipdb; ipdb.set_trace()
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(), # [20, 80, 95], 完整的梅尔谱, target
            mask.unsqueeze(1), # [20, 1, 95] 1 for value and 0 for pad
            h.transpose(1, 2).contiguous(), # [20, 80, 95], output of encoder of speech tokens
            embedding, # [20, 80] speaker embedding 
            cond=conds # [20, 80, 95] 
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        #import ipdb; ipdb.set_trace()
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder( # NOTE
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        #import ipdb; ipdb.set_trace()
        super().__init__()
        self.input_size = input_size # 512
        self.output_size = output_size # 80
        self.decoder_conf = decoder_conf # {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1, 'cfm_params': {'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}, 'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64, 'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}
        self.mel_feat_conf = mel_feat_conf # {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050, 'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}
        self.vocab_size = vocab_size # 6561 # NOTE 这里没有三个特殊的tokens, E, S, T, 6561=81*81=3^8
        self.output_type = output_type # 'mel'
        self.input_frame_rate = input_frame_rate # 25
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size) # (6561, 512) NOTE (1)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size) # Linear(in_features=192, out_features=80, bias=True) NOTE (2)
        self.encoder = encoder # <class 'cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder'> NOTE (3)
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size) # (512, 80) NOTE (4)
        self.decoder = decoder # <class 'cosyvoice.flow.flow_matching.CausalConditionalCFM'> NOTE (5) and (5) in total
        self.only_mask_loss = only_mask_loss # True
        self.token_mel_ratio = token_mel_ratio # 2 TODO for what?
        self.pre_lookahead_len = pre_lookahead_len # 3 TODO for what?

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        #import ipdb; ipdb.set_trace() # NOTE forward of CausalMaskedDiffWithXvec
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        #import ipdb; ipdb.set_trace()
        loss, _ = self.decoder.compute_loss( # NOTE
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding, # speaker embedding, extracted from ref voice
                  streaming,
                  finalize):
        #import ipdb; ipdb.set_trace()
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1) # [1, 192] -> [1, 192]
        embedding = self.spk_embed_affine_layer(embedding) # [1,192] -> Linear(192, 80) -> [1,80]

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len # [1, 87] + [1, 311] -> [1, 398]
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask # [1,398] -> Embedding(6561, 512) -> [1, 398, 512], prompt_len + text_len = 87 + 311 = 398

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming) # NOTE h.shape=[1, 796, 512] NOTE up sampling conformer encoder layers
        else:
            token, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(token, token_len, context=context, streaming=streaming)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1] # 174, 796-174=622
        h = self.encoder_proj(h) # [1, 796, 512] -> Linear(512, 80) -> [1, 796, 80]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype) # [1, 174+622, 80]
        conds[:, :mel_len1] = prompt_feat # 前面的174个位置放的是prompt feat of mel-spectrogram
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h) # [1, 796] all 1
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(), # [1, 80, 796]
            mask=mask.unsqueeze(1), # [1, 1, 796]
            spks=embedding, # [1, 80]
            cond=conds, # [1, 80, 796]
            n_timesteps=10,
            streaming=streaming # False
        ) # NOTE CausalConditionalCFM feat.shape=[1, 80, 796] # NOTE 只有这里是按照时间循环了10次的！
        feat = feat[:, :, mel_len1:] # [1, 80, 622] ? NOTE 难道说，这是622个位置的梅尔谱，一次成型了？？？ 只需要十次loop (t=0 to 1 with [0, ..., 1] totally 11 timepoints) -> 是的，是一次成型了!
        assert feat.shape[2] == mel_len2
        return feat.float(), None # [1, 80, 622]

