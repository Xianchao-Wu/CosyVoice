from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from matcha.hifigan.models import feature_loss, generator_loss, discriminator_loss
from cosyvoice.utils.losses import tpr_loss, mel_loss
# TPR = truncated Pointwise Relativistic? TODO

class HiFiGan(nn.Module):
    def __init__(self, generator, discriminator, mel_spec_transform,
                 multi_mel_spectral_recon_loss_weight=45, feat_match_loss_weight=2.0,
                 tpr_loss_weight=1.0, tpr_loss_tau=0.04):
        import ipdb; ipdb.set_trace()
        super(HiFiGan, self).__init__()
        self.generator = generator # <class 'cosyvoice.hifigan.generator.HiFTGenerator'>
        self.discriminator = discriminator # <class 'cosyvoice.hifigan.discriminator.MultipleDiscriminator'>
        self.mel_spec_transform = mel_spec_transform # [functools.partial(<function mel_spectrogram at 0x7fd03302e710>, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)]
        self.multi_mel_spectral_recon_loss_weight = multi_mel_spectral_recon_loss_weight # 45
        self.feat_match_loss_weight = feat_match_loss_weight # 2.0
        self.tpr_loss_weight = tpr_loss_weight # 1.0
        self.tpr_loss_tau = tpr_loss_tau # 0.04

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        import ipdb; ipdb.set_trace() # for training hifigan NOTE
        if batch['turn'] == 'generator':
            return self.forward_generator(batch, device)
        else:
            return self.forward_discriminator(batch, device)

    def forward_generator(self, batch, device):
        import ipdb; ipdb.set_trace()
        real_speech = batch['speech'].to(device) # torch.Size([20, 24576])
        pitch_feat = batch['pitch_feat'].to(device) # torch.Size([20, 96])
        # 1. calculate generator outputs
        generated_speech, generated_f0 = self.generator(batch, device) # <class 'cosyvoice.hifigan.generator.HiFTGenerator'>; [20, 24576] and [20, 96]
        # 2. calculate discriminator outputs
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech)
        # 3. calculate generator losses, feature loss, mel loss, tpr losses [Optional]
        loss_gen, _ = generator_loss(y_d_gs) # NOTE
        loss_fm = feature_loss(fmap_rs, fmap_gs) # NOTE
        loss_mel = mel_loss(real_speech, generated_speech, self.mel_spec_transform) # NOTE
        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_gs, y_d_rs, self.tpr_loss_tau)
        else:
            loss_tpr = torch.zeros(1).to(device)
        loss_f0 = F.l1_loss(generated_f0, pitch_feat)
        loss = loss_gen + self.feat_match_loss_weight * loss_fm + \
            self.multi_mel_spectral_recon_loss_weight * loss_mel + \
            self.tpr_loss_weight * loss_tpr + loss_f0
        return {'loss': loss, 'loss_gen': loss_gen, 'loss_fm': loss_fm, 'loss_mel': loss_mel, 'loss_tpr': loss_tpr, 'loss_f0': loss_f0}

    def forward_discriminator(self, batch, device):
        import ipdb; ipdb.set_trace()
        real_speech = batch['speech'].to(device) # [20, 24576] 最原始的语音张量
        # 1. calculate generator outputs
        with torch.no_grad():
            generated_speech, generated_f0 = self.generator(batch, device) # [20, 24576] and [20, 96]
        # 2. calculate discriminator outputs
        import ipdb; ipdb.set_trace()
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(real_speech, generated_speech.detach())
        # 3. calculate discriminator losses, tpr losses [Optional]
        import ipdb; ipdb.set_trace()
        loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs) # NOTE
        if self.tpr_loss_weight != 0:
            loss_tpr = tpr_loss(y_d_rs, y_d_gs, self.tpr_loss_tau) # NOTE
        else:
            loss_tpr = torch.zeros(1).to(device)
        loss = loss_disc + self.tpr_loss_weight * loss_tpr
        # TODO 两个loss的数量级差别比较大啊... todo
        # tensor(7.7790, device='cuda:0', grad_fn=<AddBackward0>) for loss_disc and tensor(3.2887e-05, device='cuda:0', grad_fn=<AddBackward0>) for loss_tpr
        return {'loss': loss, 'loss_disc': loss_disc, 'loss_tpr': loss_tpr}
        '''
        ipdb> loss
        tensor(7.7790, device='cuda:0', grad_fn=<AddBackward0>)
        2025-09-13 11:29:32,376 DEBUG Using selector: EpollSelector
        ipdb> loss_disc
        tensor(7.7790, device='cuda:0', grad_fn=<AddBackward0>)
        2025-09-13 11:29:34,565 DEBUG Using selector: EpollSelector
        ipdb> loss_tpr
        tensor(3.2887e-05, device='cuda:0', grad_fn=<AddBackward0>)
        '''

