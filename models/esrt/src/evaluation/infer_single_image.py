import argparse, torch, cv2, numpy as np
from src.models.esrt_max import ESRT_MAX
from src.evaluation.test_tta import tta_forward

def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR); 
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_tensor(img):
    return torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)

def to_numpy(t):
    t = t.squeeze(0).clamp(0,1).cpu().numpy().transpose(1,2,0)
    return (t*255.0 + 0.5).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--inp', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = ESRT_MAX().to(device).eval()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if 'ema' in ckpt: model.load_state_dict(ckpt['ema'], strict=False)
    else:             model.load_state_dict(ckpt['model'], strict=False)

    lr = read_rgb(args.inp)
    lr_t = to_tensor(lr).to(device)
    with torch.no_grad():
        sr_t = tta_forward(model, lr_t)
    sr = to_numpy(sr_t)
    cv2.imwrite(args.out, cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
    print('Saved:', args.out)

if __name__=='__main__': main()
