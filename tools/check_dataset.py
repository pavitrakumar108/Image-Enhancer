import os, glob, argparse, os

def list_images(folder):
    exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.webp"]
    files=[]
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

def check_root(root, scale=4):
    print("="*80); print(f"Checking dataset: {root}"); print("="*80)
    hr_dir=os.path.join(root,"HR"); lr_dir=os.path.join(root,"LR_bicubic",f"X{scale}")
    if not os.path.exists(hr_dir): print(f"Missing {hr_dir}"); return
    if not os.path.exists(lr_dir): print(f"Missing {lr_dir}"); return
    hr_files=list_images(hr_dir); lr_files=list_images(lr_dir)
    print(f"HR: {len(hr_files)} | LR: {len(lr_files)}")
    hr_ids={os.path.splitext(os.path.basename(f))[0] for f in hr_files}
    lr_ids={os.path.splitext(os.path.basename(f))[0].replace(f"x{scale}", "") for f in lr_files}
    matches=hr_ids & lr_ids
    print(f"Matched: {len(matches)}")

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--df2k_root', default='datasets/DF2K'); ap.add_argument('--scale', type=int, default=4); args=ap.parse_args()
    for subset in ['DIV2K','Flickr2K','OST']:
        p=os.path.join(args.df2k_root, subset)
        if os.path.exists(p): check_root(p, args.scale)
        else: print('Missing subset', p)
