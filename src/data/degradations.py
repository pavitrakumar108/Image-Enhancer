import random, numpy as np, cv2

def rnd_degradation(hr: np.ndarray, scale: int) -> np.ndarray:
    # Real-ISR style: blur -> downsample -> jpeg -> noise (light).
    h, w = hr.shape[:2]
    if random.random() < 0.7:
        k = random.choice([3,5,7])
        hr = cv2.GaussianBlur(hr, (k,k), sigmaX=random.uniform(0.2, 2.0))
    method = random.choice([cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
    lr = cv2.resize(hr, (w//scale, h//scale), interpolation=method)
    if random.random() < 0.5:
        enc = cv2.imencode('.jpg', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60,95)])[1]
        lr = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    if random.random() < 0.5:
        sigma = random.uniform(0, 5.0)
        noise = np.random.normal(0, sigma, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return lr
