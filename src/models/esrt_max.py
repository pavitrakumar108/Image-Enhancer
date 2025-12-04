import torch
import torch.nn as nn
from .hpb_blocks import conv3x3, PixelShuffleUpsampler, HPB, GateFuse
from .transformer_et import LTB
from .kernel_code_cnn import KernelCode


class ESRT_MAX(nn.Module):
    def __init__(
        self, 
        scale=4, 
        in_ch=3, 
        out_ch=3, 
        c=64, 
        num_hpb=5,
        et_blocks=2, 
        heads=12, 
        split=2, 
        unfold_k=3, 
        code_dim=64
    ):
        super().__init__()
        self.scale = scale

        # kernel code extractor
        self.code = KernelCode(code_dim=code_dim)

        # shallow feature extraction
        self.head = conv3x3(in_ch, c)

        # HPBs
        self.hpbs = nn.ModuleList([
            HPB(c, hfm_k=2, repeats=5, code_dim=code_dim)
            for _ in range(num_hpb)
        ])

        # fuse HPB outputs
        self.concat_reduce = nn.Conv2d(num_hpb * c, c, 1)
        self.ltb_f = LTB(
            channels=c,
            et_blocks=et_blocks,
            heads=heads,
            split=split,
            k=unfold_k
        )

        # second transformer path
        self.expand = nn.Conv2d(c, 2 * c, 1)
        self.ltb_s = LTB(
            channels=2 * c,
            et_blocks=et_blocks,
            heads=heads,
            split=split,
            k=unfold_k
        )
        self.reduce = nn.Conv2d(2 * c, c, 1)

        # gate fusion
        self.fuser = GateFuse(c)

        # ---- UPSCALING ----
        # 2× upscaler
        self.up2 = PixelShuffleUpsampler(2, c)
        self.head2 = conv3x3(c, out_ch)

        # 4× (2× + 2×)
        self.up4_a = PixelShuffleUpsampler(2, c)
        self.up4_b = PixelShuffleUpsampler(2, c)
        self.head4 = conv3x3(c, out_ch)

    def forward(self, x):

        kcode = self.code(x)

        f0 = self.head(x)

        # run HPBs
        feats = []
        f = f0
        for h in self.hpbs:
            f = h(f, kcode)
            feats.append(f)

        ff = torch.cat(feats, dim=1)
        ff = self.concat_reduce(ff)
        ff = self.ltb_f(ff)

        fs = self.expand(f0)
        fs = self.ltb_s(fs)
        fs = self.reduce(fs) + f0

        fused = self.fuser(ff, fs)

        # 2× output
        y2 = self.head2(self.up2(fused))

        # 4× output = 2× + 2×
        f4 = self.up4_b(self.up4_a(fused))
        y4 = self.head4(f4)

        return y2, y4
