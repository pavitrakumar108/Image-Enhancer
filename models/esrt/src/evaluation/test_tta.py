import torch
def _flips(x, op):
    if op==0: return x
    if op==1: return torch.flip(x, dims=[-1])
    if op==2: return torch.flip(x, dims=[-2])
    if op==3: return torch.rot90(x, 1, dims=(-2,-1))
    if op==4: return torch.rot90(x, 2, dims=(-2,-1))
    if op==5: return torch.rot90(x, 3, dims=(-2,-1))
    if op==6: return torch.flip(torch.rot90(x,1,dims=(-2,-1)), dims=[-1])
    if op==7: return torch.flip(torch.rot90(x,1,dims=(-2,-1)), dims=[-2])
    return x
@torch.no_grad()
def tta_forward(model, lr):
    outs=[]
    for op in range(8):
        inp = _flips(lr, op)
        y2, y4 = model(inp)
        y = y4
        y = _flips(y, (8-op)%8)
        outs.append(y)
    return torch.stack(outs, dim=0).mean(dim=0)
