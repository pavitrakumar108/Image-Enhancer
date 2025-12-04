import argparse, torch, yaml
from tqdm import tqdm
from src.data.df2k_loader import make_loader
from src.models.esrt_max import ESRT_MAX
from src.evaluation.test_tta import tta_forward
from src.training.loss_functions import to_y

def psnr(a,b):
    mse = torch.mean((a-b)**2); return 10.0*torch.log10(1.0/(mse+1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_cfg', default='configs/datasets/df2k_paths.yaml')
    ap.add_argument('--model_cfg', default='configs/model/esrt_max_x4.yaml')
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    mcfg = yaml.safe_load(open(args.model_cfg))
    dcfg = yaml.safe_load(open(args.data_cfg))

    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = ESRT_MAX(scale=mcfg['scale'], c=mcfg['channels'], num_hpb=mcfg['num_hpb'],
                     et_blocks=mcfg['transformer_blocks'], heads=mcfg['heads'], split=mcfg['split']).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    if 'ema' in ckpt: model.load_state_dict(ckpt['ema'], strict=False)
    else:             model.load_state_dict(ckpt['model'], strict=False)

    loader = make_loader(dcfg['train_roots'], mcfg['scale'], 'val', 96, 1, 2, degrade_mix=False, shuffle=False)
    psnrs=[]
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc='Eval-TTA'):
            lr, hr = lr.to(device), hr.to(device)
            y4 = tta_forward(model, lr).clamp(0,1)
            psnrs.append(psnr(to_y(y4), to_y(hr)).item())
    print(f'PSNR-Y (EMA + TTAx8): {sum(psnrs)/len(psnrs):.3f} dB')

if __name__=='__main__': main()
