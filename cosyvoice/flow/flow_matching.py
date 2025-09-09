# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM # NOTE 所以说，最初的basic conditional flow matching class是来自另外一个package啊！厉害了.
from cosyvoice.utils.common import set_all_random_seed


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        import ipdb; ipdb.set_trace()
        super().__init__(
            n_feats=in_channels, # 240
            cfm_params=cfm_params, # {'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}
            n_spks=n_spks, # 1
            spk_emb_dim=spk_emb_dim, # 80
        )
        self.t_scheduler = cfm_params.t_scheduler # 'cosine'
        self.training_cfg_rate = cfm_params.training_cfg_rate # 0.2
        self.inference_cfg_rate = cfm_params.inference_cfg_rate # 0.7
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0) # 240 + 80, spk=speaker
        # Just change the architecture of the estimator here
        self.estimator = estimator # <class 'cosyvoice.flow.decoder.CausalConditionalDecoder'>

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        import ipdb; ipdb.set_trace()
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        import ipdb; ipdb.set_trace()
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x # [1, 80, 796] -> [2, 80, 796]
            mask_in[:] = mask # [1, 1, 796] -> [2, 1, 796]
            mu_in[0] = mu # [1, 80, 796] -> [2, 80, 796]
            t_in[:] = t.unsqueeze(0) # shape=[2], value=[0, 0]
            spks_in[0] = spks # [1, 80]
            cond_in[0] = cond # [1, 80, 796]
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            ) # dphi/dt NOTE dphi_dt.shape=[2, 80, 796]
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0) # [1, 80, 796] and [1, 80, 796]
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt) # Equation (14) in cosyvoice-1 paper: 2407.05407v2; beta=self.inference_cfg_rate=0.7 NOTE -> [1, 80, 796]
            x = x + dt * dphi_dt # [1, 80, 796] + 0.0123 * [1, 80, 796] -> [1, 80, 796]
            t = t + dt # 0 + 0.0123 -> 0.0123
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float() # [1, 80, 796]

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming) # NOTE here, type of self.estimator = <class 'cosyvoice.flow.decoder.CausalConditionalDecoder'>, x.shape=[2,80,796], mask.shape=[2,1,796], mu.shape=[2,80,796], t_in.shape=[2], spks_in.shape=[2,80], cond_in.shape=[2,80,796], streaming=False   --->  [2, 80, 796]
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                estimator.set_input_shape('x', (2, 80, x.size(2)))
                estimator.set_input_shape('mask', (2, 1, x.size(2)))
                estimator.set_input_shape('mu', (2, 80, x.size(2)))
                estimator.set_input_shape('t', (2,))
                estimator.set_input_shape('spks', (2, 80))
                estimator.set_input_shape('cond', (2, 80, x.size(2)))
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             x.data_ptr()]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        import ipdb; ipdb.set_trace()
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1) # z=epsilon=noise=X_0 ~ N(0, I_d)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1 # 论文中的公式(11)
        u = x1 - (1 - self.sigma_min) * z # NOTE 如果u和t无关，就有意思了，毕竟t是刚随机采样出来的，x1不可能和t相关. 这里，u是构造出来的target，让NN的预测结果和u的mse loss最小化.

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        import ipdb; ipdb.set_trace()
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        # in_channels: 240
        # cfm_params: {'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}
        # 1
        # 80
        # <class 'cosyvoice.flow.decoder.CausalConditionalDecoder'>
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300]) # NOTE what is 50, 300 for? 类似于最长噪声长度，也是能够处理的speech token sequence的最长长度; 就是开个最大长度，然后可复用即可。 frame rate of 50 Hz

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps) [1, 80, 796], logic is [1, 87/prompt_speech_token+311/text_2_qwen2lm_speech_token] -> input_embedding (6561,512) -> self.encoder/up-sample conformer encoder style -> [1, 796, 512] -> self.encoder_proj Linear(512, 80) -> [1, 796, 80] -> [1, 80, 796] 
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps) [1, 1, 796], all 1
            n_timesteps (int): number of diffusion steps 10
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None. 
                shape: (batch_size, spk_emb_dim) [1, 80]
            cond: Not used but kept for future purposes [1, 80, 796], [1, 80, 174/prompt_ref_voice_mel_spectrogram+622/'0']

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        import ipdb; ipdb.set_trace()
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature # [1, 80, 15000] -> cut -> [1, 80, 796], temperature=1.0
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype) # [0.0, 0.1, ..., 1.0] 一共11个元素, 0.0 to 1.0 之间的11个点; torch.Size([11])=t_span.shape
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi) # tensor([0.0000, 0.0123, 0.0489, 0.1090, 0.1910, 0.2929, 0.4122, 0.5460, 0.6910, 0.8436, 1.0000], device='cuda:0')
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None # NOTE --> [1, 80, 796] 这是执行了10次不同timestamp下的结果，一次欧拉采样，生成一个next mel-spectrogram

