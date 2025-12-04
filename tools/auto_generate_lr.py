import os, argparse
from PIL import Image

def bicubic_down(hr_path, out_path, scale=4):
    img=Image.open(hr_path).convert('RGB'); w,h=img.size
    lr=img.resize((w//scale,h//scale), resample=Image.BICUBIC)
    os.makedirs(os.path.dirname(out_path), exist_ok=True); lr.save(out_path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hr_dir', required=True); ap.add_argument('--out_dir', required=True); ap.add_argument('--scale', type=int, default=4); args=ap.parse_args()
    for root,_,files in os.walk(args.hr_dir):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                hrp=os.path.join(root,f); base=os.path.splitext(f)[0]
                outp=os.path.join(args.out_dir,f"{base}x{args.scale}.png"); bicubic_down(hrp,outp,args.scale); print('Saved', outp)
if __name__=='__main__': main()
