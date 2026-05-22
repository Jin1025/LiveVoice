# Vendored from Microsoft UniSpeech speaker_verification (ECAPA-TDNN head).
# https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans


def _load_s3prl_upstream(feat_type: str):
    """Load WavLM/HuBERT upstream without torch.hub (avoids broken hub imports)."""
    if feat_type == "wavlm_large":
        from s3prl.upstream.wavlm.hubconf import wavlm_large

        return wavlm_large()
    if feat_type == "wavlm_base_plus":
        from s3prl.upstream.wavlm.hubconf import wavlm_base_plus

        return wavlm_base_plus()
    if feat_type in ("hubert_large_ll60k", "hubert_large"):
        from s3prl.upstream.hubert.hubconf import hubert_large_ll60k

        return hubert_large_ll60k()
    if feat_type == "wav2vec2_xlsr":
        from s3prl.upstream.wav2vec2.hubconf import wav2vec2_xlsr

        return wav2vec2_xlsr()
    return torch.hub.load("s3prl/s3prl", feat_type, trust_repo=True)


class Res2Conv1dReluBn(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, f"{channels} % {scale} != 0"
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        feat_type="fbank",
        sr=16000,
        feature_selection="hidden_states",
        update_extract=False,
        config_path=None,
    ):
        super().__init__()
        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type in ("fbank", "mfcc"):
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == "fbank":
            self.feature_extract = trans.MelSpectrogram(
                sample_rate=sr,
                n_fft=512,
                win_length=win_len,
                hop_length=hop_len,
                f_min=0.0,
                f_max=sr // 2,
                pad=0,
                n_mels=feat_dim,
            )
        elif feat_type == "mfcc":
            melkwargs = {
                "n_fft": 512,
                "win_length": win_len,
                "hop_length": hop_len,
                "f_min": 0.0,
                "f_max": sr // 2,
                "pad": 0,
            }
            self.feature_extract = trans.MFCC(
                sample_rate=sr, n_mfcc=feat_dim, log_mels=False, melkwargs=melkwargs
            )
        else:
            if config_path is not None:
                raise NotImplementedError(
                    "fairseq upstream checkpoints (unispeech_sat) are not bundled; "
                    "use wavlm_large / wavlm_base_plus via s3prl."
                )
            self.feature_extract = _load_s3prl_upstream(feat_type)
            enc = getattr(getattr(self.feature_extract, "model", None), "encoder", None)
            if enc is not None and len(enc.layers) == 24:
                for idx in (11, 23):
                    sa = getattr(enc.layers[idx].self_attn, "fp32_attention", None)
                    if sa is not None:
                        enc.layers[idx].self_attn.fp32_attention = False

        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type not in ("fbank", "mfcc"):
            freeze_list = [
                "final_proj",
                "label_embs_concat",
                "mask_emb",
                "project_q",
                "quantizer",
            ]
            for name, param in self.feature_extract.named_parameters():
                if any(fv in name for fv in freeze_list):
                    param.requires_grad = False
            if not self.update_extract:
                for param in self.feature_extract.parameters():
                    param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        device = next(self.feature_extract.parameters()).device
        wav = [torch.randn(self.sr, device=device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        return 1

    def get_feat(self, x):
        if self.update_extract:
            features = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type in ("fbank", "mfcc"):
                    features = self.feature_extract(x) + 1e-6
                else:
                    features = self.feature_extract([sample for sample in x])

        if self.feat_type == "fbank":
            x = features.log()
        elif self.feat_type in ("mfcc",):
            x = features
        else:
            x = features[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = (
                F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        return self.instance_norm(x)

    def forward(self, x):
        x = self.get_feat(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        return self.linear(out)


def ECAPA_TDNN_SMALL(
    feat_dim,
    emb_dim=256,
    feat_type="fbank",
    sr=16000,
    feature_selection="hidden_states",
    update_extract=False,
    config_path=None,
):
    return ECAPA_TDNN(
        feat_dim=feat_dim,
        channels=512,
        emb_dim=emb_dim,
        feat_type=feat_type,
        sr=sr,
        feature_selection=feature_selection,
        update_extract=update_extract,
        config_path=config_path,
    )
