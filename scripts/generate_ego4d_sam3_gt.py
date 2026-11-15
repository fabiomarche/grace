from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import pandas as pd
import spacy
import csv
from tqdm import tqdm
from moviepy import *
import sam3
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
import gc

"""
To run this script please make a sam3_env install everything needed and run the script with the following line:
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python scripts/generate_ego4d_sam3_gt.py.

Remember: SAM3 allows to segment objects for each frame given a specific prompt. Thus, the idea is to give 
SAM3 the grace concept present inside the clip text. 

We can divide this script into four phases:
    1. Obtaining the effective concepts: for each clip we extract from the dataset the 'clip_text' column and we lemmatize it 
    thanks to a very light nlp model (in simpler words: we transform every word into its base form, e.g. running --> run, 
    apples --> apple). Finally, we take the lemmatized version and intersect it with the whole set of the grace vocabulary.
    2. SAM3 inference: we run SAM3 passing it the clip and the respective prompt (which might be several words, thus we iterate
    the inference for every concept)
    3. File writing: two files will be built at the end of the script, a csv file representing the ground truth dataset and several .npz files containing the binary 
    mask (hence the segmentation mask) for each object.
    Starting with the .csv file, we compose the dataset appending the result for each frame. 
    The features of the dataset are: |id_sample|frame_idx|object_name|object_id|is_present|confidence|
    Where:
    -  'id_sample' is a string given by "{video_uid}:{clip_start:.2f}-{clip_end:.2f}"
    - 'frame_idx' is an integer and corresponds to the index of the frame.
    - 'object_name' is a string and it's the grace concept (i.e. the prompt passed for the relative step to SAM3)
    - 'object_idx' is an integer corresponding to the id assigned by SAM3 to the object. In other words: if there are multiple entities of the same concept
    SAM3 makes a different mask for each of them
    - 'is_present' is a boolean representing whether the object appears during that frame or not
    - 'confidence' is a float representing how confident is SAM3 of its output

    (id_sample, frame_idx, object_name, object_id) is the primary key.

    Regarding the .npz files, there is a .npz file for each pair clip-object. Inside of each of them there are the masks for each frame
"""

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
    """
    It extracts the grace concepts appearing in the clip text. We are using sets and not lists
    because they allow O(1) access.
    """
    doc = nlp(clip_text.lower())

    clip_nouns_set = set([token.lemma_ for token in doc if token.pos_ == "NOUN"])

    valid_objects = grace_concepts.intersection(clip_nouns_set)

    return list(valid_objects)
    
def main(args):
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    autocast_ctx.__enter__()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ROOT_CLIPS = args.rootGraceData / "ego4d" / "clips"
    GT_CSV_PATH = args.rootGraceData / "ego4d" / "benchmark" / f"ego4d_sam3_gt_{args.typeVocab}.csv"
    MASKS_GT_PATH = args.rootGraceData / "ego4d" / "benchmark" / f"sam3_gt_masks_{args.typeVocab}"
    
    if not os.path.exists(ROOT_CLIPS):
        raise FileNotFoundError(f"There is no folder with path: {ROOT_CLIPS}")

    if not os.path.exists(MASKS_GT_PATH):
        os.makedirs(MASKS_GT_PATH)

    # Instead of using pandas which would require us to either mantain a huge element in RAM or keep building and destroying small dataframes,
    # we use the csv writer
    with open(GT_CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_sample", "frame_idx", "object_name", "object_id", "is_present", "confidence"])
    
    grace_concepts = get_grace_concepts(typeVocab=args.typeVocab)
    benchmark_df = get_df_benchmark(root=args.rootGraceData)

    nlp = spacy.load("en_core_web_sm")

    sam_predictor = build_sam3_multiplex_video_predictor(
        use_fa3=False,
        use_rope_real=False,
    )

    sam_predictor.model.forward_backbone_per_frame_for_eval = True
    sam_predictor.model.trim_past_non_cond_mem_for_eval = True
    sam_predictor.model.offload_output_to_cpu_for_eval = True

    if hasattr(sam_predictor.model, "tracker"):
        sam_predictor.model.tracker.forward_backbone_per_frame_for_eval = True
        sam_predictor.model.tracker.trim_past_non_cond_mem_for_eval = True
        sam_predictor.model.tracker.offload_output_to_cpu_for_eval = True

    original_init = sam_predictor.model.init_state
    def patched_init_state(**kwargs):
        # It silently removes the parameter that make the model crash
        kwargs.pop("offload_state_to_cpu", None)

        # It mantains the frames in RAM instead of occupying the GPU.
        # The frame will be moved in VRAM only when necessary
        kwargs["offload_video_to_cpu"] = True 
        return original_init(**kwargs)
    sam_predictor.model.init_state = patched_init_state

    # We set the batch of the frames passed to SAM3 to 4 (instead of 16).
    # This allows to have a lighter (but slower) VRAM occupation
    sam_predictor.model.batched_grounding_batch_size = 4
    if hasattr(sam_predictor.model, "tracker"):
        sam_predictor.model.tracker.batched_grounding_batch_size = 4


    # It frees the old chunk of frames before analyzing the new one
    orig_batched_vg = sam_predictor.model.detector.forward_video_grounding_batched_multigpu
    def patched_batched_vg(*args, **kwargs):
        kwargs["batch_size"] = 4
        g_cache = kwargs.get("grounding_cache")
        f_idx = kwargs.get("frame_idx", 0)
        if g_cache is not None and "grounding_buffer" in g_cache:
            buf = g_cache["grounding_buffer"]
            # If the current frame belongs to a new chunk it clears the VRAM
            if buf and all(not (k[0] <= f_idx < k[1]) for k in buf.keys()):
                buf.clear()
                torch.cuda.empty_cache()
        return orig_batched_vg(*args, **kwargs)

    sam_predictor.model.detector.forward_video_grounding_batched_multigpu = patched_batched_vg

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

                valid_objects = extract_valid_objects(
                    nlp=nlp,
                    clip_text=row.clip_text,
                    grace_concepts=grace_concepts
                )

                
                sample_id = f"{name_group}:{clip_start}-{clip_end}"

                for prompt in valid_objects:
                    # Since a clip might have several objects (hence several prompts) we must reset the current session
                    response = sam_predictor.handle_request(
                        request=dict(
                            type="start_session",
                            resource_path=clip_path,
                        )
                    )

                    session_id = response["session_id"]

                    _ = sam_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=str(prompt)
                        )
                    )

                    csv_rows = []
                    masks_dict = {}

                    # handle_stream_request() allows to serialize the responses. In other words:
                    # if we'd have used a standard function, SAM3 would process all n frames, and we'd
                    # accumulate n dictionaries and lists (unfeasible). Instead, thanks to this approach
                    # as soon as SAM3 ends the computation for the i-th frame it immediately delivers it.
                    # We can now work on single dictionary and list, and then we can free them and pass to 
                    # the next frame
                    for stream_resp in sam_predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                        )
                    ):
                        frame_idx = stream_resp["frame_index"]
                        frame_out = stream_resp["outputs"]
                        
                        obj_ids = frame_out["out_obj_ids"]
                        probs = frame_out["out_probs"]
                        masks = frame_out["out_binary_masks"]

                        if hasattr(obj_ids, "cpu"): obj_ids = obj_ids.cpu().numpy()
                        if hasattr(probs, "cpu"): probs = probs.cpu().numpy()
                        if hasattr(masks, "cpu"): masks = masks.cpu().numpy()

                        if len(obj_ids) == 0: # no object detected in this frame
                            csv_rows.append([sample_id, frame_idx, prompt, -1, False, 0.0])
                        else:
                            # Remember that for each frame, SAM3 might found several entities for the same prompt
                            for obj_idx, obj in enumerate(obj_ids):
                                mask_i = masks[obj_idx].astype(bool)
                                is_present = bool(mask_i.any())
                                conf = float(probs[obj_idx]) if is_present else 0.0

                                csv_rows.append([sample_id, frame_idx, prompt, int(obj), is_present, round(conf,4)])

                            if masks.any():
                                masks_dict[f"frame_{frame_idx}"] = masks.astype(bool)

                        # We free the memory
                        del frame_out, obj_ids, probs, masks
                    
                    with open(GT_CSV_PATH, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_rows)

                    mask_file_path = MASKS_GT_PATH / f"{name_group}_{clip_start}-{clip_end}_{prompt}.npz"
                    np.savez_compressed(mask_file_path, **masks_dict)
                    

                    # This frees the VRAM before entering the next clip
                    _ = sam_predictor.handle_request(
                        request=dict(type="close_session", session_id=session_id)
                    )
                    del csv_rows, masks_dict

                    # We are forcing the garbage collector to immediately clean the RAM.
                    # It must be called before torch.cuda.empty_cache() because it might happen
                    # that an object in RAM (hence in CPU) keeps a hidden ref with a tensor in VRAM (hence in GPU).
                    # This is a problem because until that object is not cleaned by the gc, pytorch refuses to free
                    # the VRAM
                    gc.collect()
                    # We free the VRAM before passing to the next prompt
                    torch.cuda.empty_cache()


                pbar.update(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)