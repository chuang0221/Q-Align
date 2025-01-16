import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import argparse
from tqdm import tqdm
from PIL import Image
import warnings
import sys

from q_align import QAlignVideoScorer, load_video

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class WarningSuppress:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

def run_vqa_prediction(scorer, video_path):
    """Run VQA prediction using Q-Align"""
    try:
        video_list = [load_video(video_path)]
        scores = scorer(video_list).tolist()
        return scores[0]  # Return first score since we process one video at a time
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def evaluate_vqa_model(videos_dir, mos_file, device="cuda:0"):
    """Evaluate Q-Align VQA model against MOS scores"""
    # Initialize Q-Align scorer
    scorer = QAlignVideoScorer(device=device)
    
    sheet_name = None
    if "hdr2sdr" in videos_dir.lower():
        sheet_name = "HDR2SDR"
    elif "sdr" in videos_dir.lower():
        sheet_name = "SDR"
    
    if sheet_name is None:
        raise ValueError("Could not determine sheet_name from videos_dir. Path should contain 'hdr2sdr' or 'sdr'")

    mos_df = pd.read_excel(mos_file, sheet_name=sheet_name)
    
    results = []
    
    for _, row in tqdm(mos_df.iterrows(), total=len(mos_df), desc=f"Processing {sheet_name} videos"):
        video_name = row['vid']
        mos_score = row['mos']
        
        video_path = os.path.join(videos_dir, video_name)
        video_path = video_path + '.mp4'
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        pred_score = run_vqa_prediction(scorer, video_path)
        if pred_score is not None:
            results.append({
                'video_name': video_name,
                'mos': mos_score,
                'predicted_score': pred_score * 5  # Scale to 0-5 range
            })

    results_df = pd.DataFrame(results)

    srocc = spearmanr(results_df['mos'], results_df['predicted_score'])[0]
    plcc = pearsonr(results_df['mos'], results_df['predicted_score'])[0]

    timestamp = pd.Timestamp.now().strftime('%Y%m%d')
    results_dir = os.path.join(os.path.dirname(__file__), 
                              'qalign_experiment', 
                              f'short-{sheet_name.lower()}',
                              'Q-Align',
                              timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nResults for Q-Align on {sheet_name} sheet:")
    print(f"SROCC: {srocc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    
    results_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"SROCC: {srocc:.4f}\n")
        f.write(f"PLCC: {plcc:.4f}\n\n")
        
        results_df['abs_diff'] = abs(results_df['predicted_score'] - results_df['mos'])
        best_5 = results_df.nsmallest(5, 'abs_diff')
        worst_5 = results_df.nlargest(5, 'abs_diff')
        
        f.write("Top 5 Best Predictions:\n")
        f.write(best_5[['video_name', 'mos', 'predicted_score', 'abs_diff']].to_string())
        f.write("\n\nTop 5 Worst Predictions:\n")
        f.write(worst_5[['video_name', 'mos', 'predicted_score', 'abs_diff']].to_string())
    
    return srocc, plcc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--mos_file', type=str, required=True,
                       help='Path to Excel file containing MOS scores')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run Q-Align on (e.g., cuda:0, cpu)')
    
    args = parser.parse_args()
    
    evaluate_vqa_model(args.videos_dir, args.mos_file, args.device)

if __name__ == "__main__":
    main()