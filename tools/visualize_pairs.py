import argparse, os, random
from PIL import Image
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hr_dir', required=True); ap.add_argument('--lr_dir', required=True); args=ap.parse_args()
    hrs=[os.path.join(args.hr_dir,f) for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp'))]
    if not hrs: print('No HR images'); return
    hrp=random.choice(hrs); base=os.path.splitext(os.path.basename(hrp))[0]
    candidates=[os.path.join(args.lr_dir,f) for f in os.listdir(args.lr_dir) if f.startswith(base)]
    if not candidates: print('No LR match for', base); return
    lrp=candidates[0]; hr=Image.open(hrp).convert('RGB'); lr=Image.open(lrp).convert('RGB')
    fig,ax=plt.subplots(1,2,figsize=(8,4)); ax[0].imshow(lr); ax[0].set_title('LR'); ax[0].axis('off'); ax[1].imshow(hr); ax[1].set_title('HR'); ax[1].axis('off'); plt.show()
if __name__=='__main__': main()
