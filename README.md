# image-to-video-creation-minor-prohect
# ==========================================================
# Stable Video Diffusion - T4 v3
# Longer clips (chained segments), sharper output (Real-ESRGAN),
# smoother motion, better video encoding.
# Paste into a Colab cell (GPU runtime: T4)
# ==========================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "transformers", "accelerate", "safetensors",
                "opencv-python", "imageio", "imageio-ffmpeg"], check=True)

import gc, os, re, urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from google.colab import files

# ---------------- SETTINGS ----------------
DESCRIPTION     = "A person walking through a rainy city"
MOTION_STRENGTH = 3            # 1-5
CAMERA          = "static"     # static / pan / zoom / tracking
FPS             = 7            # 6 or 7 (conditioning for the model)
FRAMES          = 14           # 14 (img2vid) or 25 (img2vid-xt)
SEED            = 42

# ---- duration ----
SEGMENTS        = 3            # clips chained end-to-start. 1 = old behaviour (~2s), 3 = ~5s, 4 = ~6.5s
CROSSFADE       = 3            # frames blended at each join
INTERP_PASSES   = 1            # 0 = none, 1 = 2x frames, 2 = 4x frames (smoother)
SLOW_MO         = 1.0          # below 1.0 = slower and longer (0.8 = 25% longer)
LOOP            = False        # play forward then backward

# ---- clarity ----
GEN_RES         = (1024, 576)  # SVD native size; falls back to 768x432 automatically if VRAM runs out
OUTPUT_RES      = (1280, 720)  # final video size
STEPS           = 30           # diffusion steps (default is 25)
REAL_ESRGAN     = True         # AI upscaler; falls back to Lanczos + sharpen if it can't load
CLARITY_BOOST   = True         # gentle local-contrast boost

# ---- stability ----
AUTO_MOTION_BOOST = True       # +1 motion if description says run/chase/sprint
DEFLICKER         = True       # evens out brightness/colour pulsing
ANTI_DRIFT        = True       # keeps colour/sharpness from degrading across segments
# ------------------------------------------

assert torch.cuda.is_available(), "Enable a GPU runtime (T4)"
assert FRAMES in (14, 25) and FPS in (6, 7) and 1 <= MOTION_STRENGTH <= 5
assert CAMERA in ("static", "pan", "zoom", "tracking")
assert SEGMENTS >= 1 and 0 <= INTERP_PASSES <= 2 and 0 <= CROSSFADE < FRAMES - 1


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------- Real-ESRGAN (self-contained, no extra pip packages) ----------------
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def load_sr():
    if not REAL_ESRGAN:
        return None
    try:
        url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-general-x4v3.pth")
        path = "realesr-general-x4v3.pth"
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        state = torch.load(path, map_location="cpu")
        state = state.get("params_ema", state.get("params", state))
        net = SRVGGNetCompact()
        net.load_state_dict(state, strict=True)
        net = net.eval().half().cuda()
        print("Real-ESRGAN loaded.")
        return net
    except Exception as e:
        print(f"Real-ESRGAN unavailable ({e}); using Lanczos + sharpen instead.")
        return None


@torch.no_grad()
def sr_frame(net, img, out_size):
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).half().cuda() / 255.0
    y = net(x).clamp_(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    out = Image.fromarray((y * 255).round().astype(np.uint8))
    return out if out.size == tuple(out_size) else out.resize(tuple(out_size), Image.LANCZOS)


# ---------------- Source image ----------------
def generate_still(description, seed):
    from diffusers import AutoPipelineForText2Image
    t2i = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    img = t2i(
        prompt=description + ", cinematic wide shot, natural lighting, sharp focus, highly detailed",
        num_inference_steps=4, guidance_scale=0.0, height=432, width=768,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    del t2i
    free_memory()
    return img


def get_source():
    print("Upload a starting image, or cancel the dialog to generate one from DESCRIPTION.")
    try:
        up = files.upload()
    except Exception:
        up = {}
    if up:
        return Image.open(list(up.keys())[0]).convert("RGB")
    print("No upload - generating a still from the description...")
    return generate_still(DESCRIPTION, SEED)


def make_conditioning(src, size, sr_net):
    """Crop to 16:9 (no stretching) and upscale small sources with the AI upscaler."""
    if sr_net is not None and (src.width < size[0] or src.height < size[1]):
        src = sr_frame(sr_net, src, (src.width * 4, src.height * 4))
    return ImageOps.fit(src, size, Image.LANCZOS)


# ---------------- Helpers for stability ----------------
def match_stats(img, ref, strength=0.4):
    a = np.asarray(img).astype(np.float32)
    r = np.asarray(ref).astype(np.float32)
    ma, sa = a.mean((0, 1)), a.std((0, 1)) + 1e-3
    mr, sr_ = r.mean((0, 1)), r.std((0, 1)) + 1e-3
    matched = (a - ma) / sa * sr_ + mr
    out = a * (1 - strength) + matched * strength
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def anti_drift(frame, ref):
    f = match_stats(frame, ref, 0.4)
    return f.filter(ImageFilter.UnsharpMask(radius=1.5, percent=40, threshold=2))


def join_segments(segs, overlap):
    out = list(segs[0])
    for s in segs[1:]:
        s = s[1:]                      # first frame duplicates the conditioning frame
        n = min(overlap, len(s), len(out))
        for i in range(n):
            out[-n + i] = Image.blend(out[-n + i], s[i], (i + 1) / (n + 1))
        out += s[n:]
    return out


def deflicker(frames, strength=0.7, window=7):
    arrs = [np.asarray(f).astype(np.float32) for f in frames]
    means = np.array([a.mean(axis=(0, 1)) for a in arrs])
    pad = window // 2
    padded = np.pad(means, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smooth = np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=1)
    out = []
    for a, m, s in zip(arrs, means, smooth):
        gain = 1 + strength * (s / np.maximum(m, 1e-3) - 1)
        out.append(Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8)))
    return out


def interpolate(frames):
    """One optical-flow midpoint frame between each pair (2x frame count)."""
    out = []
    w, h = frames[0].size
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    gx, gy = gx.astype(np.float32), gy.astype(np.float32)
    for a, b in zip(frames, frames[1:]):
        A, B = np.asarray(a), np.asarray(b)
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 21, 3, 5, 1.1, 0)
        wa = cv2.remap(A, gx - 0.5 * flow[..., 0], gy - 0.5 * flow[..., 1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(B, gx + 0.5 * flow[..., 0], gy + 0.5 * flow[..., 1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out += [a, Image.fromarray(cv2.addWeighted(wa, 0.5, wb, 0.5, 0))]
    out.append(frames[-1])
    return out


def apply_camera(frames, mode, amount=0.15):
    """Simulated camera move by cropping (not true 3D camera motion)."""
    if mode == "static":
        return frames
    n = len(frames)
    w, h = frames[0].size
    out = []
    for i, f in enumerate(frames):
        t = i / (n - 1)
        if mode == "zoom":
            scale, px = 1 + amount * t, 0.5
        elif mode == "pan":
            scale, px = 1 + amount, t
        else:  # tracking: slow push-in plus slight sideways drift
            scale, px = 1 + amount * t, 0.5 + 0.25 * t
        cw, ch = w / scale, h / scale
        x0 = px * (w - cw)
        y0 = (h - ch) / 2
        out.append(f.resize((w, h), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch)))
    return out


def clarity(img, amount=0.5, clip=1.5):
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    l = lab[..., 0]
    boosted = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    lab[..., 0] = cv2.addWeighted(l, 1 - amount, boosted, amount, 0)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def enhance(frames, size, sr_net):
    out = []
    for i, f in enumerate(frames):
        if sr_net is not None:
            g = sr_frame(sr_net, f, size)
        else:
            g = f.resize(size, Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 60, 3))
        if CLARITY_BOOST:
            g = clarity(g)
        out.append(g)
        if (i + 1) % 20 == 0:
            print(f"  enhanced {i + 1}/{len(frames)}")
    return out


def write_video(frames, path, fps):
    try:
        import imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=None,
                                pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_params=["-crf", "16", "-preset", "slow"]) as wr:
            for f in frames:
                wr.append_data(np.asarray(f))
    except Exception as e:
        print(f"High-quality encoder failed ({e}); using default encoder.")
        export_to_video(frames, path, fps=fps)


# ==================== RUN ====================
free_memory()
sr_net = load_sr()
src = get_source()

words = set(re.findall(r"[a-z]+", DESCRIPTION.lower()))
if AUTO_MOTION_BOOST and words & {"run", "runs", "running", "chase", "chasing", "sprint", "sprinting"}:
    MOTION_STRENGTH = min(5, MOTION_STRENGTH + 1)
MOTION_BUCKETS = {1: 40, 2: 80, 3: 127, 4: 170, 5: 220}

image = make_conditioning(src, GEN_RES, sr_net)
image.save("source_frame.png")
ref = image

model_id = ("stabilityai/stable-video-diffusion-img2vid-xt" if FRAMES == 25
            else "stabilityai/stable-video-diffusion-img2vid")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()


def run_svd(img, seed, size):
    return pipe(
        image=img, height=size[1], width=size[0],
        num_frames=FRAMES, fps=FPS,
        motion_bucket_id=MOTION_BUCKETS[MOTION_STRENGTH],
        noise_aug_strength=0.02, num_inference_steps=STEPS,
        decode_chunk_size=2,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).frames[0]


print(f"Segment 1/{SEGMENTS} at {GEN_RES[0]}x{GEN_RES[1]}...")
try:
    seg = run_svd(image, SEED, GEN_RES)
except torch.cuda.OutOfMemoryError:
    print("Out of memory - falling back to 768x432.")
    free_memory()
    GEN_RES = (768, 432)
    image = make_conditioning(src, GEN_RES, sr_net)
    ref = image
    seg = run_svd(image, SEED, GEN_RES)
segments = [seg]

for k in range(1, SEGMENTS):
    print(f"Segment {k + 1}/{SEGMENTS}...")
    cond = segments[-1][-1]
    if ANTI_DRIFT:
        cond = anti_drift(cond, ref)
    segments.append(run_svd(cond, SEED + k, GEN_RES))
    free_memory()

del pipe
free_memory()

# ---------- post-processing ----------
print("Post-processing...")
frames = join_segments(segments, CROSSFADE)
if DEFLICKER:
    frames = deflicker(frames)
for _ in range(INTERP_PASSES):
    frames = interpolate(frames)
frames = apply_camera(frames, CAMERA)
frames = enhance(frames, OUTPUT_RES, sr_net)
if LOOP:
    frames = frames + frames[-2:0:-1]

playback_fps = max(6, round(FPS * (2 ** INTERP_PASSES) * SLOW_MO))
write_video(frames, "output_video.mp4", playback_fps)
print(f"{len(frames)} frames at {playback_fps} fps = {len(frames) / playback_fps:.1f}s "
      f"({OUTPUT_RES[0]}x{OUTPUT_RES[1]})")
files.download("output_video.mp4")
