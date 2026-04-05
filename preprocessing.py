import os
import pandas as pd
from tqdm import tqdm

def preprocess_mimic_cxr(root_dir, output_csv="./data/mimic_preprocessed.csv"):
    """
    root_dir: path to MIMIC-CXR images & reports (dicom/png + associated text)
    output_csv: CSV path with 'file_path', 'report_text'
    """
    data = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".png") or file.endswith(".jpg"):
                file_path = os.path.join(dirpath, file)
                # Assume report text is in .txt file with same basename
                txt_file = file_path.replace(".png", ".txt").replace(".jpg", ".txt")
                if os.path.exists(txt_file):
                    with open(txt_file, "r") as f:
                        report_text = f.read().strip()
                    data.append({"file_path": file_path, "report_text": report_text})
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed CSV saved: {output_csv}")

if __name__ == "__main__":
    mimic_root = "/path/to/mimic-cxr"
    preprocess_mimic_cxr(mimic_root)