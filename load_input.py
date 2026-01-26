import json
import os
import glob
import shutil
from PIL import Image # 需要先 pip install pillow，通常环境里都有

# ================= 核心配置区 =================
# 1. 你的原始照片文件夹
SOURCE_IMG_DIR = "./my_photo"   

# 2. 场景名称 (作为ID)
SCENE_NAME = "custom_scene_01"  

# 3. 数据集根目录
DATA_ROOT = "./data"

# 4. 拆分逻辑: 前几张做输入(Context)，剩下的做预测(Target)
# RayZer要求: num_views = num_input + num_target
NUM_CONTEXT = 4 
# ============================================

def generate_rayzer_data():
    # --- 1. 准备目录结构 ---
    scene_dir = os.path.join(DATA_ROOT, SCENE_NAME)
    images_dir = os.path.join(scene_dir, "images")
    
    if os.path.exists(scene_dir):
        print(f"[清理] 删除旧数据: {scene_dir}")
        shutil.rmtree(scene_dir)
    os.makedirs(images_dir, exist_ok=True)

    # --- 2. 收集图片 ---
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(SOURCE_IMG_DIR, ext)))
    img_files.sort() # 排序保证顺序一致
    
    total_imgs = len(img_files)
    if total_imgs == 0:
        print(f"❌ 错误: {SOURCE_IMG_DIR} 是空的！")
        return

    print(f"[处理] 发现 {total_imgs} 张图片...")

    # --- 3. 生成符合 Dataloader 要求的 JSON ---
    frames_list = []
    
    for i, src_path in enumerate(img_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(images_dir, filename)
        shutil.copy(src_path, dst_path)
        
        # 读取真实图片尺寸，防止宽高比拉伸
        with Image.open(src_path) as img:
            w, h = img.size

        # 构造单帧数据 (必须包含 fx, fy, cx, cy, w2c)
        frames_list.append({
            "file_path": f"images/{filename}",
            # 即使是 Unposed，代码也会读这些值来做 Resize，所以必须给个初值
            # 这里假设焦距很大(FOV小)，中心在正中
            "fx": float(w),  
            "fy": float(w),
            "cx": w / 2.0,
            "cy": h / 2.0,
            "w2c": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            # 额外的字段防止其他地方报错
            "K": [[float(w), 0.0, w/2.0], [0.0, float(w), h/2.0], [0.0, 0.0, 1.0]],
            "img_size": [w, h]
        })

    # 封装完整 JSON
    cameras_json = {
        "scene_name": SCENE_NAME,     # 【关键】__getitem__ 会读这个
        "frames": frames_list,        # 【关键】__getitem__ 会读这个
        "scale": 1.0,
        "center": [0.0, 0.0, 0.0]
    }
    
    json_path = os.path.join(scene_dir, "opencv_cameras.json")
    with open(json_path, "w") as f:
        json.dump(cameras_json, f, indent=4)

    # --- 4. 生成 dataset list txt ---
    txt_path = os.path.join(DATA_ROOT, "my_dataset_list.txt")
    with open(txt_path, "w") as f:
        # 写入 json 的路径
        f.write(json_path)

    # --- 5. 生成 view index json ---
    # 确保不越界
    num_input = min(NUM_CONTEXT, total_imgs - 1)
    if num_input < 1: num_input = 1 # 至少一张输入
    
    num_target = total_imgs - num_input
    
    input_indices = list(range(num_input))
    target_indices = list(range(num_input, total_imgs))
    
    view_idx_data = {
        SCENE_NAME: {
            "context": input_indices,
            "target": target_indices
        }
    }
    
    idx_json_path = os.path.join(DATA_ROOT, "my_view_index.json")
    with open(idx_json_path, "w") as f:
        json.dump(view_idx_data, f, indent=4)

    print("\n✅ 数据准备完成！")
    print("-" * 60)
    print("请直接复制并运行以下指令 (参数已自动对齐):")
    
    # 自动生成指令
    cmd = (
        f"torchrun --nproc_per_node 8 --nnodes 1 \\\n"
        f"    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \\\n"
        f"    inference.py --config \"configs/rayzer_dl3dv.yaml\" \\\n"
        f"    training.dataset_path=\"{txt_path}\" \\\n"
        f"    training.batch_size_per_gpu=1 \\\n"
        f"    training.target_has_input=false \\\n"
        f"    training.num_views={total_imgs} \\\n"         # 动态填入总数
        f"    training.num_input_views={num_input} \\\n"    # 动态填入输入数
        f"    training.num_target_views={num_target} \\\n"  # 动态填入目标数
        f"    inference.if_inference=true \\\n"
        f"    inference.compute_metrics=false \\\n"
        f"    inference.render_video=true \\\n"
        f"    inference.view_idx_file_path=\"{idx_json_path}\" \\\n"
        f"    inference.model_path=./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \\\n"
        f"    inference_out_root=./experiments/my_test"
    )
    print(cmd)
    print("-" * 60)

if __name__ == "__main__":
    generate_rayzer_data()