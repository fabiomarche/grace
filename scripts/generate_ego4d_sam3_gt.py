from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import pandas as pd
import spacy
from tqdm import tqdm
from moviepy import *

def parse_args(args = sys.argv[1:]):
    parser = ArgumentParser()
    parser.add_argument("--rootGraceData", type=Path, default=Path(os.environ.get("GRACE_DATA", "grace_data")), help="data root (default: $GRACE_DATA, else grace_data/)")
    parser.add_argument("--typeVocab", type=str, choices=["standard", "full"], default="standard", help="it defines whether to use the standard vocab or the full")
    return parser.parse_args(args)

def get_grace_concepts(typeVocab: str) -> list:
    vocab_name = "egoclip_verbs_nouns.txt" if typeVocab == "standard" else "egoclip_mined_top25_verbs_nouns.txt"
    vocab_path = f"data/concepts_vocs/video/{vocab_name}"

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"The file with the path {vocab_path} does not exist!")

    with open(vocab_path, "r") as f:
        concepts = set(f.read().splitlines())

    return concepts

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

def extract_valid_objects(nlp, clip_text: str, grace_concepts: set) -> list:
    doc = nlp(clip_text.lower())

    clip_nouns_set = set([token.lemma_ for token in doc if token.pos_ == "NOUN"])

    valid_objects = grace_concepts.intersection(clip_nouns_set)

    return list(valid_objects)

def main(args):
    ROOT_CLIPS = args.rootGraceData / "ego4d" / "clips"
    
    if not os.path.exists(ROOT_CLIPS):
        raise FileNotFoundError(f"There is no folder with path: {ROOT_CLIPS}")
    
    grace_concepts = get_grace_concepts(typeVocab=args.typeVocab)
    benchmark_df = get_df_benchmark(root=args.rootGraceData)

    nlp = spacy.load("en_core_web_sm")

    with tqdm(total=len(benchmark_df), desc="Generating GT", unit="clip") as pbar:
        for name_group, group_df_uid in benchmark_df.groupby(by="video_uid"):
            clip_video_folder = f"{ROOT_CLIPS}/{name_group}"
            if not os.path.exists(clip_video_folder):
                raise FileNotFoundError(f"There is no folder with path: {clip_video_folder}")
    
            for row in group_df_uid.itertuples(index=False):
                clip_start = f"{row.clip_start:.2f}".replace(".","_")
                clip_end = f"{row.clip_end:.2f}".replace(".","_")

                clip_path = f"{clip_video_folder}/{clip_start}-{clip_end}.mp4"

                if not os.path.exists(clip_path):
                    raise FileNotFoundError(f"There is no such file: {clip_path}")

                clip = VideoFileClip(clip_path)
                valid_objects = extract_valid_objects(nlp, clip_text=row.clip_text, grace_concepts=grace_concepts)
                # TODO chiamata a SAM3

                pbar.update(1)
                clip.close() # freeing the RAM

if __name__ == "__main__":
    args = parse_args()
    main(args)