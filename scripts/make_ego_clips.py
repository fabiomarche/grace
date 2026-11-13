from pathlib import Path
import pandas as pd
import os
from argparse import ArgumentParser
import sys
from moviepy import * 
from tqdm import tqdm

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

            original_video_path = f"{ROOT_VIDEOS}/{name_group}.mp4"
            original_video = VideoFileClip(original_video_path)

            clip_video_folder_path = f"{ROOT_CLIPS}/{name_group}"
            if not os.path.exists(clip_video_folder_path):
                os.mkdir(clip_video_folder_path)

            # .itertuples() is much faster wrt .iterrows()
            for row in group_df_uid.itertuples(index=False):
                clip_start = f"{row.clip_start:.2f}".replace(".", "_")
                clip_end = f"{row.clip_end:.2f}".replace(".", "_")
                clip_name = f"{clip_start}-{clip_end}.mp4"

                clip_path = f"{clip_video_folder_path}/{clip_name}"

                if os.path.exists(clip_path): # we skip this clip because it's already there 
                    pbar.update(1)
                    continue

                safe_end = min(row.clip_end, original_video.duration) # df is a bit dirty, so thsi is essential

                with original_video.subclipped(row.clip_start, safe_end) as sub:
                    sub.write_videofile(clip_path, audio=False, logger=None) # the audio is useless

                pbar.update(1)

            original_video.close() # it frees the RAM

    print("All clips have been generated!")

if __name__ == "__main__":
    args = parse_args()
    main(args)