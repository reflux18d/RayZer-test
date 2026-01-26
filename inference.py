# Copyright (c) 2025 Hanwen Jiang. Created for the RayZer project.

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F

def load_images_from_folder(folder_path, config, device):
    """
    [终极复刻版 V2] 包含 Config 自动同步功能
    """
    print(f"🚀 [Direct Mode] Loading images from: {folder_path}")
    
    # --- 1. 收集图片 ---
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))
    img_files.sort()
    
    if len(img_files) == 0:
        raise ValueError(f"No images found in {folder_path}")

    # 读取配置参数
    resize_h = config.model.image_tokenizer.image_size
    patch_size = config.model.image_tokenizer.patch_size
    square_crop = config.training.get("square_crop", False)
    
    # --- 2. 核心预处理 ---
    images = []
    intrinsics = []
    
    for img_path in img_files:
        image = Image.open(img_path).convert('RGB')
        original_image_w, original_image_h = image.size
        
        resize_w = int(resize_h / original_image_h * original_image_w)
        resize_w = int(round(resize_w / patch_size) * patch_size)
        
        image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
        
        start_h, start_w = 0, 0
        if square_crop:
            min_size = min(resize_h, resize_w)
            start_h = (resize_h - min_size) // 2
            start_w = (resize_w - min_size) // 2
            image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))
            
        img_np = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        
        fx = resize_w
        fy = resize_w 
        cx = resize_w / 2.0
        cy = resize_h / 2.0
        
        if square_crop:
            cx -= start_w
            cy -= start_h
            
        fxfycxcy = torch.tensor([fx, fy, cx, cy]).float()
        
        images.append(img_tensor)
        intrinsics.append(fxfycxcy)

    # --- 3. Padding 解决尺寸不一致 ---
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    padded_images = []
    padded_intrinsics = []
    
    for img, intr in zip(images, intrinsics):
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0) 
            
        padded_images.append(img)
        padded_intrinsics.append(intr)
        
    images = torch.stack(padded_images, dim=0)       
    intrinsics = torch.stack(padded_intrinsics, dim=0) 

    # --- 4. 伪造位姿与归一化 ---
    c2ws_np = np.eye(4)[None, ...].repeat(len(images), axis=0)
    c2ws = torch.from_numpy(c2ws_np).float()
    
    scene_scale_factor = config.training.get("scene_scale_factor", 1.35)
    
    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    if scene_scale < 1e-6: scene_scale = 1.0 
    scene_scale = scene_scale_factor * scene_scale
    c2ws[:, :3, 3] /= scene_scale

    # --- 5. 构造 Batch 并同步 Config ---
    num_views = len(images)
    num_input = config.training.get("num_input_views", 0)
    if num_input == 0: num_input = num_views // 2
    if num_input >= num_views: num_input = num_views - 1 
    
    num_target = num_views - num_input

    # 【关键修改】强制覆盖 Config，防止后续 Model Init 报错
    try:
        # 尝试解锁 Config (针对 OmegaConf)
        from omegaconf import OmegaConf
        OmegaConf.set_struct(config, False)
    except:
        pass # 如果不是 OmegaConf 或者 import 失败，忽略

    config.training.num_views = num_views
    config.training.num_input_views = num_input
    config.training.num_target_views = num_target
    print(f"🔄 [Direct Mode] Config Synced: Total={num_views}, Input={num_input}, Target={num_target}")

    indices = torch.zeros((num_views, 2)).long()
    
    batch = {
        "image": images.to(device).unsqueeze(0),        
        "c2w": c2ws.to(device).unsqueeze(0),            
        "fxfycxcy": intrinsics.to(device).unsqueeze(0), 
        "index": indices.to(device).unsqueeze(0),
        "scene_name": [os.path.basename(folder_path)],
        "context_indices": torch.arange(num_input).long().to(device).unsqueeze(0),
        "target_indices": torch.arange(num_input, num_views).long().to(device).unsqueeze(0)
    }
    
    return [batch]
# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}


# ================= MODIFIED START =================
# 检查是否开启了“直读文件夹模式”
direct_folder = config.inference.get("direct_image_folder", None)

if direct_folder is not None and os.path.isdir(direct_folder):
    print(f"⚠️ Switching to Direct Folder Mode: {direct_folder}")
    dataloader = load_images_from_folder(direct_folder, config, ddp_info.device)
    datasampler = None  # 【重要】必须在这里定义为 None，否则后面报错
else:
    # 【模式 B】原来的逻辑 (读 JSON/TXT)
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)

    datasampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        sampler=datasampler
    )
# ================= MODIFIED END =================
dataloader_iter = iter(dataloader)

dist.barrier()

# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])


if config.inference.get("model_path", None) is not None and config.inference.get("if_inference", False):
    # use the specified checkpoint if specified
    checkpoint = torch.load(config.inference.model_path, map_location="cpu", weights_only=True)
    model.module.load_state_dict(checkpoint["model"], strict=False) # strict=False because the loss computer is different from internal version, BE CAREFUL!
elif config.inference.get("model_path", None) is None and config.inference.get("if_inference", False):
    # otherwise, use the last training checkpoint
    model.module.load_ckpt(config.training.checkpoint_dir)

inference_sampling = config.inference.view_idx_file_path.replace('.json', '').split('_')[-1]
inference_out_dir = os.path.join(
    config.get("inference_out_root", None), config.training.wandb_exp_name + '_' + config.training.view_selector.type + '_' + inference_sampling
)
if ddp_info.is_main_process:  
    print(f"Running inference; save results to: {inference_out_dir}")
    os.makedirs(inference_out_dir, exist_ok=True)
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()

if datasampler is not None:
    datasampler.set_epoch(0)
model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        if 'LVSM' in config.model.class_name:
            result = model(batch)
            result= model.module.render_video(result, **config.inference.render_video_config)
        elif 'rayzer' in config.model.class_name:
            result = model(batch, create_visual=True, render_video=config.inference.get("render_video", False))

        if hasattr(result, 'input'):
            # 无论 result.input 是 dict 还是 EasyDict，我们都尝试赋值
            try:
                if isinstance(result.input, dict):
                    result.input['scene_name'] = batch['scene_name']
                else:
                    result.input.scene_name = batch['scene_name']
            except Exception as e:
                print(f"⚠️ Warning: Failed to inject scene_name into result.input: {e}")

        # 为了保险起见，如果代码之后改了逻辑去读 result.scene_name，我们也保留上一版的补丁
        if isinstance(result, dict):
            result['scene_name'] = batch['scene_name']
        else:
            result.scene_name = batch['scene_name']
        export_results(result, inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))

    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)