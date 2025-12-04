import torch, torch.nn as nn

class EMHA(nn.Module):
    def __init__(self, dim, heads=12, split=2):
        super().__init__()
        assert dim % 2 == 0
        self.dim=dim; self.heads=heads; self.split=split
        self.reduce = nn.Linear(dim, dim//2)
        self.qkv = nn.Linear(dim//2, 3*(dim//2))
        self.proj = nn.Linear(dim//2, dim)
    def forward(self, x):
        B,N,C = x.shape; C1=C//2
        xr = self.reduce(x)
        qkv = self.qkv(xr).view(B,N,3,self.heads,C1//self.heads)
        q,k,v = qkv.unbind(dim=2)
        q=q.permute(0,2,1,3); k=k.permute(0,2,1,3); v=v.permute(0,2,1,3)
        assert N % self.split==0
        seg = N//self.split; scale = (k.shape[-1]) ** -0.5
        outs=[]
        for i in range(self.split):
            qs=q[:,:,i*seg:(i+1)*seg,:]; ks=k[:,:,i*seg:(i+1)*seg,:]; vs=v[:,:,i*seg:(i+1)*seg,:]
            attn=(qs @ ks.transpose(-2,-1))*scale
            outs.append(attn.softmax(dim=-1) @ vs)
        o=torch.cat(outs, dim=2).permute(0,2,1,3).contiguous().view(B,N,C1)
        return self.proj(o)

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hid=int(dim*ratio)
        self.fc1=nn.Linear(dim,hid); self.fc2=nn.Linear(hid,dim)
    def forward(self,x):
        import torch.nn.functional as F
        return self.fc2(F.gelu(self.fc1(x)))

class ETBlock(nn.Module):
    def __init__(self, dim, heads=12, split=2):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=EMHA(dim,heads,split)
        self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim)
    def forward(self,x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class LTB(nn.Module):
    def __init__(self, channels, et_blocks=2, heads=12, split=2, k=3):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=k, padding=k//2, stride=1)
        dim = channels*(k*k)
        self.blocks = nn.ModuleList([ETBlock(dim, heads, split) for _ in range(et_blocks)])
        self.fold=None; self.k=k
    def forward(self,x):
        B,C,H,W = x.shape
        tokens = self.unfold(x).transpose(1,2)
        for b in self.blocks: tokens=b(tokens)
        tokens = tokens.transpose(1,2)
        if self.fold is None:
            self.fold=nn.Fold(output_size=(H,W), kernel_size=self.k, padding=self.k//2, stride=1)
        return self.fold(tokens)
