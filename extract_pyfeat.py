import os
import pandas as pd
from feat import Detector

print("--- Initializing Py-Feat Detector for Intel Mac (CPU) ---")
# Initialize the Py-Feat detector for CPU execution
# Using default models for face detection, facial landmarks, action units, and emotions
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="xgb",
    emotion_model="resmasknet",
    device="cpu"
)

def extract_video_features(participant_id, robot_trials):
    # NOTE: IndividualAnalysis.py now does this extraction inline (with ffmpeg
    # conversion, confidence filtering, and AU-column resolution) and caches the
    # result, so you likely don't need to run this script separately at all. It's
    # left here, with the filename fixed, in case you want a standalone extraction
    # pass (e.g. to run unattended overnight before doing any merging).
    for robot in robot_trials:
        # MasterScriptTest.py records webcam footage as participant_1_Robot1_cam.avi,
        # not participant_1_Robot1.mp4 -- this was pointing at a file that never gets created.
        video_path = f"participant_{participant_id}_{robot}_cam.avi"
        output_csv = f"participant_{participant_id}_{robot}_openface.csv"
        
        if not os.path.exists(video_path):
            print(f"  [!] Video not found: {video_path}. Skipping...")
            continue
            
        print(f"  [>] Processing video: {video_path}...")
        
        # Run Py-Feat detection on the video
        # Note: skip_frames can be adjusted if you want to speed up processing (e.g., skip_frames=2)
        predictions = detector.detect_video(video_path, skip_frames=1)
        
        # Py-Feat returns a Fex object. We convert it to a pandas DataFrame.
        df_feats = predictions.to_df()
        
        # Ensure the timestamp column matches what your main pipeline expects
        # Py-Feat typically outputs 'timestamp' or 'frame'. Let's normalize it to 'timestamp'.
        if "timestamp" not in df_feats.columns and "frame" in df_feats.columns:
            # If timestamp isn't explicit, estimate from frame rate or keep frame as timestamp
            df_feats["timestamp"] = df_feats["frame"] / 30.0  # Assuming ~30 fps default if needed
            
        # Keep relevant columns (timestamp, success indicator if available, and Action Units like AU12_r)
        if "success" not in df_feats.columns:
            df_feats["success"] = 1
            
        # Save to the CSV filename your integration script looks for
        df_feats.to_csv(output_csv, index=False)
        print(f"  [+] Successfully generated real Py-Feat features: {output_csv}")

if __name__ == "__main__":
    try:
        p_id = int(input("Enter participant ID to extract video features for: "))
    except ValueError:
        print("Invalid ID.")
        exit()
        
    ROBOT_TRIALS = ["Robot1", "Robot2", "Robot3", "Robot4"]
    extract_video_features(p_id, ROBOT_TRIALS)
    print("\nExtraction complete! You can now run your main integration script.")