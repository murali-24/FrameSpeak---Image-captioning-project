import pandas as pd
from collections import defaultdict

def load_and_process_captions(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, delimiter="|")

    # Clean and preprocess
    df[" comment"] = df[" comment"].str.lower().str.strip()
    df = df[df[" comment"].notna()]
    df = df[df[" comment"] != ""]
    df = df.drop_duplicates(subset=["image_name", " comment"])

    # Create caption dictionary
    caption_dict = defaultdict(list)
    for _, row in df.iterrows():
        caption_dict[row["image_name"]].append(row[" comment"])
    
    return caption_dict