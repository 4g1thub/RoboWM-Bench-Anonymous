import os

import cv2
from tqdm import tqdm


def get_crop_region(src_w, src_h, dst_w, dst_h, src_cfg, dst_cfg, h_shift=0.8, v_shift=0.5):
    tan_half_fov_src = src_cfg["aperture"] / (2 * src_cfg["focal"])
    tan_half_fov_dst = dst_cfg["aperture"] / (2 * dst_cfg["focal"])

    scale_factor = tan_half_fov_dst / tan_half_fov_src
    crop_w = int(src_w * scale_factor)

    target_aspect = dst_w / dst_h
    crop_h = int(crop_w / target_aspect)

    if crop_h > src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)

    max_x_offset = src_w - crop_w
    max_y_offset = src_h - crop_h

    x1 = int(max_x_offset * h_shift)
    y1 = int(max_y_offset * v_shift)

    return x1, y1, crop_w, crop_h


def collect_video_tasks(input_root):
    lerobot_tasks = []
    fallback_tasks = []

    for root, _, files in os.walk(input_root):
        mp4_files = [file for file in files if file.endswith(".mp4")]
        if not mp4_files:
            continue

        for file in mp4_files:
            full_path = os.path.join(root, file)
            fallback_tasks.append(full_path)
            if "observation.images.top_rgb" in root:
                lerobot_tasks.append(full_path)

    if lerobot_tasks:
        print(f"Detected LeRobot directory structure. Found {len(lerobot_tasks)} top_rgb videos.")
        return sorted(lerobot_tasks)

    if fallback_tasks:
        print(f"No top_rgb subdirectory detected. Falling back to all mp4 files under the input directory. Found {len(fallback_tasks)} videos.")
        return sorted(fallback_tasks)

    return []


def process_all_videos(input_root, output_root):
    src_cfg = {"focal": 1.4139, "aperture": 2.7700}
    dst_cfg = {"focal": 28.7, "aperture": 38.11}
    src_res = (1280, 720)
    dst_res = (640, 480)

    x1, y1, cw, ch = get_crop_region(
        src_res[0],
        src_res[1],
        dst_res[0],
        dst_res[1],
        src_cfg,
        dst_cfg,
        h_shift=0.65,
        v_shift=0.5,
    )

    print(f"Crop window: start=({x1}, {y1}), size={cw}x{ch}")

    video_tasks = collect_video_tasks(input_root)

    if not video_tasks:
        print("No matching video files found. Please check the input path.")
        return

    for video_path in tqdm(video_tasks, desc="Processing"):
        rel_path = os.path.relpath(video_path, input_root)
        save_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_path, fourcc, fps, dst_res)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[y1 : y1 + ch, x1 : x1 + cw]
            resized = cv2.resize(cropped, dst_res, interpolation=cv2.INTER_LANCZOS4)

            out.write(resized)

        cap.release()
        out.release()


if __name__ == "__main__":
    # Adjust these relative paths for your local video directory layout.
    INPUT_PATH = "./videos"
    OUTPUT_PATH = "./videos_resized"

    process_all_videos(INPUT_PATH, OUTPUT_PATH)
    print("All videos have been processed.")
