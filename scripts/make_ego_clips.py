from pathlib import Path
import pandas as pd
import os
from argparse import ArgumentParser
import sys
from moviepy import * 
from tqdm import tqdm
import cv2
from PIL import Image
import shutil

"""
This script consists of taking the already downloaded video, and clipping them
based on the ground truth.
To allow a better management and a clean structure, this script does the following:
- it creates the folder 'clips' under the 'grace_data/ego4d' folder
- it creates a folder for every video. The name of the folder is the uid of the video
- it saves the single clips inside the respective video. The name of each clip is given by
'startClip-EndClip.mp4', where both startClip and endClip are formatted taking only 2 decimals.

This script allows to not waste basically any computation if it already finds the clips.
"""

def parse_args(args = sys.argv[1:]):
    parser =ArgumentParser()
    parser.add_argument("--rootGraceData", type=Path, default=Path(os.environ.get("GRACE_DATA", "grace_data")), help="data root (default: $GRACE_DATA, else grace_data/)")
    return parser.parse_args(args)

def get_df_benchmark(root: Path) -> pd.DataFrame:
    benchmark_path = f"{root}/ego4d/benchmark/ego4d_benchmark_clips.csv"

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"The file with the path {benchmark_path} does not exist!")
    
    df = pd.read_csv(
        filepath_or_buffer=benchmark_path,
        sep="\t",
        usecols=range(10),
    )

    return df

def extract_images_from_mp4(mp4_path: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    idx=0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(out_dir / f"{idx:06d}.jpg", quality=95)
        idx+=1
    cap.release()
    return idx

def main(args):
    df_benchmark = get_df_benchmark(root=args.rootGraceData)

    ROOT_VIDEOS = args.rootGraceData / "ego4d" / "videos" / "v2" / "full_scale"
    
    if not os.path.exists(ROOT_VIDEOS):
        raise FileNotFoundError(f"The folder with the path: {ROOT_VIDEOS} is not there. Please download the videos before running this script")
    
    ROOT_CLIPS = args.rootGraceData / "ego4d" / "clips"

    if not os.path.exists(ROOT_CLIPS):
        os.makedirs(ROOT_CLIPS)

    with tqdm(total=len(df_benchmark), desc="Cutting clips", unit="clip") as pbar:
        # .groupby() is the native way to aggregate data
        for name_group, group_df_uid in df_benchmark.groupby(by="video_uid"):

            clip_video_folder_path = ROOT_CLIPS / str(name_group)
            clip_video_folder_path.mkdir(parents=True, exist_ok=True)

            original_video = None

            # .itertuples() is much faster wrt .iterrows()
            for row in group_df_uid.itertuples(index=False):
                clip_start = f"{row.clip_start:.2f}".replace(".", "_")
                clip_end = f"{row.clip_end:.2f}".replace(".", "_")
                clip_name = f"{clip_start}-{clip_end}"
                
                clip_folder_path = clip_video_folder_path / clip_name
                clip_folder_path.mkdir(parents=True, exist_ok=True)

                legacy_mp4_path = clip_video_folder_path / f"{clip_name}.mp4"
                clip_mp4_path = clip_folder_path / f"{clip_name}.mp4"

                # If older .mp4 exist in the parent folder, we move it in the sub folder
                if legacy_mp4_path.exists() and not clip_mp4_path.exists():
                    shutil.move(str(legacy_mp4_path), str(clip_mp4_path))

                if not clip_mp4_path.exists():
                    if original_video is None:
                        original_video_path = ROOT_VIDEOS / f"{name_group}.mp4"
                        original_video = VideoFileClip(str(original_video_path))

                    safe_end = min(row.clip_end, original_video.duration) # df is a bit dirty, so thsi is essential
                    with original_video.subclipped(row.clip_start, safe_end) as sub:
                        sub.write_videofile(str(clip_mp4_path), audio=False, logger=None) # the audio is useless

                if not any(clip_folder_path.glob("*.jpg")):
                    extract_images_from_mp4(mp4_path=clip_mp4_path, out_dir=clip_folder_path)

                pbar.update(1)

            if original_video is not None:
                original_video.close() # it frees the RAM

    print("All clips have been generated!")

if __name__ == "__main__":
    args = parse_args()
    main(args)