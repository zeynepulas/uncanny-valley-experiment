import pandas as pd
import os
import subprocess
from feat import Detector

# ==========================================
# CONFIGURATION
# ==========================================
PARTICIPANT_ID = 1
ROBOT_TRIALS = ["Robot1", "Robot2", "Robot3", "Robot4"]

# You only need these four AUs (AU4, AU5, AU7, AU9). Restricting to them keeps the
# merged CSV small and makes it obvious if Py-Feat's column names don't match what
# we expect (see the column-resolution step below).
TARGET_AUS = ["AU4", "AU5", "AU7", "AU9"]

# Py-Feat's confidence column is typically "FaceScore" (0-1), not OpenFace's binary
# "success". 0.75 is the cutoff used in a published EMG-validation study comparing
# OpenFace/Py-Feat/FaceReader AU accuracy; adjust if you have a reason to.
FACE_CONFIDENCE_THRESHOLD = 0.75

# How many frames Py-Feat actually processes per second of video (source is ~30fps).
#   1  -> every frame  (~30Hz, finest detail, slowest)
#   5  -> ~6Hz         (reasonable middle ground -- current default)
#   30 -> ~1Hz         (fastest, coarsest -- what this script used before)
FRAME_SAMPLE_INTERVAL = 5

# Nominal capture fps set in MasterScriptTest.py's cv2.VideoWriter. Cross-check this
# against the "X frames written" console line MasterScriptTest.py prints for each
# trial vs. that trial's actual video_playback duration (from the PPI CSV) -- if the
# real fps drifted from 30 (common with OpenCV on macOS under load), update this.
ACTUAL_FPS = 30.0

# Max allowed gap (seconds) between a PPI sample and its nearest facial-AU sample in
# the merge below. Without this, merge_asof(direction="nearest") will happily pair
# samples that are seconds apart. Widen if too many rows end up with NaN AUs; narrow
# if matches look too loose.
MERGE_TOLERANCE_SEC = 1.0

print(f"--- Processing Individual Analysis & Py-Feat Extraction for Participant {PARTICIPANT_ID} ---")

godspeed_file = f"participant_{PARTICIPANT_ID}_godspeed_scores.csv"
ppi_file = f"participant_{PARTICIPANT_ID}_continuous_ppi.csv"
timing_log_filename = f"participant_{PARTICIPANT_ID}_camera_timing.csv"

try:
    ppi_df = pd.read_csv(ppi_file)
    scores_df = pd.read_csv(godspeed_file)
except FileNotFoundError as e:
    print(f"Error loading base files for participant {PARTICIPANT_ID}: {e}")
    exit()

# Wall-clock moment the webcam actually started recording each trial (written by the
# updated MasterScriptTest.py). Used below to correct for the gap between when the
# video_playback phase clock resets to 0 and when frame 0 of the webcam footage was
# actually captured -- these are NOT the same instant (camera init takes >1.5s).
camera_timing = {}
if os.path.exists(timing_log_filename):
    timing_df = pd.read_csv(timing_log_filename)
    camera_timing = dict(zip(timing_df["robot_name"], timing_df["camera_start_time"]))
else:
    print(f"  [!] WARNING: {timing_log_filename} not found -- facial timestamps will NOT be "
          f"corrected for the camera-start offset, and will assume frame 0 == video_playback "
          f"phase start.")

# Inject participant_id and map trial_name to robot_name for merging
ppi_df["participant_id"] = PARTICIPANT_ID
if "trial_name" in ppi_df.columns:
    ppi_df["robot_name"] = ppi_df["trial_name"]

# The video-playback phase clock starts the instant the phase begins in software,
# but VLC is launched fresh each trial (cold app start + fullscreen + file open),
# so the robot video isn't actually visible on screen for several seconds after
# that. Since the analysis only needs per-trial averages (not fine time-locked
# reactions to specific on-screen moments), the simplest fix is to drop this dead
# window rather than try to detect VLC's exact playback-start moment. Set this
# comfortably above the largest VLC launch delay you've actually observed across
# a few pilot runs -- 4-7s was seen during testing, so 8s leaves some margin, but
# re-check this once you have real participant data.
STIMULUS_ONSET_TRIM_SEC = 8.0

# Isolate video playback phase (automatically drops camera retry flag rows)
video_ppi = ppi_df[ppi_df["phase"] == "video_playback"].copy()
if "invalid_ppi" in video_ppi.columns:
    video_ppi = video_ppi[video_ppi["invalid_ppi"] == False]

before_trim = len(video_ppi)
video_ppi = video_ppi[video_ppi["elapsed_trial_sec"] >= STIMULUS_ONSET_TRIM_SEC]
print(f"  [i] Trimmed {before_trim - len(video_ppi)} PPI row(s) from the first "
      f"{STIMULUS_ONSET_TRIM_SEC}s of each trial (VLC launch dead time).")

trial_integrated_dfs = []

# Initialize Py-Feat Detector safely once globally in CPU mode.
# emotion_model is set to None since you only need AU4/5/7/9 -- if your installed
# py-feat version doesn't accept None here, this falls back to the default model.
try:
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model=None,
        device="cpu",
        n_jobs=1,
    )
    print("Initializing Py-Feat Detector (CPU mode, emotion model disabled for speed)...")
except Exception as e:
    print(f"  [!] Could not disable emotion model on this py-feat version ({e}); "
          f"falling back to the default (slower) setup.")
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        device="cpu",
        n_jobs=1,
    )

# ==========================================
# STEP A: PY-FEAT EXTRACTION & TRIAL ALIGNMENT
# ==========================================
for robot_name in ROBOT_TRIALS:
    print(f"\nProcessing trial: {robot_name}...")

    video_filename = f"participant_{PARTICIPANT_ID}_{robot_name}_cam.avi"
    expected_openface_csv = f"participant_{PARTICIPANT_ID}_{robot_name}_openface.csv"
    temp_mp4 = f"temp_{robot_name}.mp4"

    # 1. AUTOMATE PY-FEAT EXTRACTION IF CSV DOES NOT EXIST YET
    if not os.path.exists(expected_openface_csv):
        if os.path.exists(video_filename):
            print(f"  [*] Converting, downscaling to 480p, and running Py-Feat on {video_filename}...")
            try:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", video_filename,
                    "-vf", "scale=640:-1",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-g", "30",
                    "-movflags", "+faststart",
                    temp_mp4
                ], check=True)

                predictions = detector.detect_video(temp_mp4, batch_size=1, every_n_frame=FRAME_SAMPLE_INTERVAL)
                predictions.to_csv(expected_openface_csv, index=False)

                print(f"  [+] Py-Feat successfully processed {robot_name}")
            except Exception as e:
                print(f"  [!] Warning: Py-Feat extraction failed for {robot_name}: {e}")
            finally:
                if os.path.exists(temp_mp4):
                    os.remove(temp_mp4)
        else:
            print(f"  [!] Warning: Video file {video_filename} not found.")
    else:
        print(f"  [i] Found existing CSV for {robot_name}, skipping extraction.")

    # Isolate PPI data for this specific robot trial
    trial_ppi = video_ppi[video_ppi["robot_name"] == robot_name].copy()
    if trial_ppi.empty:
        print(f"  [!] Warning: No PPI data found for {robot_name}")
        continue

    if not os.path.exists(expected_openface_csv):
        print(f"  [!] Skipping facial data merge for {robot_name} (No facial output available).")
        trial_integrated_dfs.append(trial_ppi)
        continue

    # 2. LOAD, FILTER, AND MERGE ACTION UNIT DATA
    openface_df = pd.read_csv(expected_openface_csv)
    openface_df.columns = openface_df.columns.str.strip()

    # Confidence filtering. Py-Feat's confidence column is typically "FaceScore", not
    # OpenFace's "success" -- check for either so this doesn't silently no-op.
    conf_col = next((c for c in ["FaceScore", "face_score", "success"] if c in openface_df.columns), None)
    if conf_col:
        before = len(openface_df)
        threshold = 1 if conf_col == "success" else FACE_CONFIDENCE_THRESHOLD
        openface_df = openface_df[openface_df[conf_col] >= threshold]
        print(f"  [i] Dropped {before - len(openface_df)} low-confidence frame(s) using '{conf_col}'.")
    else:
        print(f"  [!] WARNING: no recognized confidence column in {expected_openface_csv}. "
              f"Actual columns: {list(openface_df.columns)}")

    # Resolve the real AU column names. Py-Feat's xgb model typically outputs bare
    # names like "AU4" rather than OpenFace-style "AU4_r" -- check both so a naming
    # mismatch doesn't silently drop your target AUs.
    resolved_au_cols = {}
    for au in TARGET_AUS:
        for candidate in (au, f"{au}_r"):
            if candidate in openface_df.columns:
                resolved_au_cols[au] = candidate
                break
    missing_aus = set(TARGET_AUS) - set(resolved_au_cols.keys())
    if missing_aus:
        print(f"  [!] WARNING: could not find column(s) for {missing_aus} in {expected_openface_csv}. "
              f"Actual columns: {list(openface_df.columns)} -- fix TARGET_AUS or the candidate "
              f"names above once you see the real output.")

    trial_ppi = trial_ppi.sort_values("elapsed_trial_sec")

    # Correct the facial timestamp using the measured camera-start offset (see
    # camera_timing above) instead of assuming frame 0 == phase start.
    camera_offset_sec = 0.0
    if robot_name in camera_timing and not trial_ppi.empty:
        phase_start_wallclock = (trial_ppi["absolute_time"] - trial_ppi["elapsed_trial_sec"]).iloc[0]
        camera_offset_sec = camera_timing[robot_name] - phase_start_wallclock
        print(f"  [i] Measured camera-start offset for {robot_name}: {camera_offset_sec:.3f}s")

    if "frame" in openface_df.columns:
        openface_df["timestamp"] = camera_offset_sec + openface_df["frame"] / ACTUAL_FPS
    elif "timestamp" in openface_df.columns:
        openface_df["timestamp"] = camera_offset_sec + openface_df["timestamp"]
    else:
        print(f"  [!] WARNING: no 'frame' or 'timestamp' column in {expected_openface_csv} -- "
              f"cannot time-align facial data. Actual columns: {list(openface_df.columns)}")
        trial_integrated_dfs.append(trial_ppi)
        continue

    openface_df = openface_df.sort_values("timestamp")

    # Keep only what we actually need from the facial side before merging
    facial_cols_to_keep = ["timestamp"] + list(resolved_au_cols.values())
    if conf_col:
        facial_cols_to_keep.append(conf_col)
    openface_slim = openface_df[facial_cols_to_keep].rename(
        columns={v: k for k, v in resolved_au_cols.items()}  # e.g. "AU4_r" -> "AU4"
    )

    merged_trial = pd.merge_asof(
        trial_ppi,
        openface_slim,
        left_on="elapsed_trial_sec",
        right_on="timestamp",
        direction="nearest",
        tolerance=MERGE_TOLERANCE_SEC,
    )

    trial_integrated_dfs.append(merged_trial)

if not trial_integrated_dfs:
    print("[!] Error: No trial data could be processed.")
    exit()

all_trials_timeseries = pd.concat(trial_integrated_dfs, ignore_index=True)

# ==========================================
# STEP B: PROCESS & MERGE GODSPEED SCORES
# ==========================================
print("\nProcessing and merging Godspeed questionnaire ratings...")

def process_godspeed_scores(scores_df, target_participant_id):
    subscales = {
        "anthropomorphism": ["anthropomorphism_1", "anthropomorphism_2", "anthropomorphism_3", "anthropomorphism_4", "anthropomorphism_5"],
        "animacy": ["animacy_1", "animacy_2", "animacy_3", "animacy_4", "animacy_5", "animacy_6"],
        "likeability": ["likeability_1", "likeability_2", "likeability_3", "likeability_4", "likeability_5"],
        "perceived_intelligence": ["intelligence_1", "intelligence_2", "intelligence_3", "intelligence_4", "intelligence_5"],
        "perceived_safety": ["safety_1", "safety_2", "safety_3"]
    }

    scores_df = scores_df.copy()

    # The filename (participant_{ID}_godspeed_scores.csv) is treated as ground truth
    # for whose data this is. If the file's own participant_id column disagrees,
    # that's almost certainly a mislabeled/misentered ID during collection -- warn
    # loudly instead of silently filtering everything out (which is what happened
    # with your test file: it's internally labeled participant_id=2).
    if "participant_id" in scores_df.columns:
        found_ids = set(scores_df["participant_id"].unique())
        if found_ids != {target_participant_id}:
            print(f"  [!] WARNING: {godspeed_file} is internally labeled with participant_id "
                  f"{found_ids}, not {target_participant_id} (from the filename). Treating all "
                  f"rows as participant {target_participant_id}'s data -- double-check this is "
                  f"actually the right participant's file.")
    scores_df["participant_id"] = target_participant_id

    processed_dfs = []
    for robot_name, group in scores_df.groupby("robot_name"):
        if len(group) > 1:
            print(f"  [!] WARNING: {len(group)} Godspeed rows found for participant "
                  f"{target_participant_id}, robot {robot_name} -- using the first one.")
        row_data = {"participant_id": target_participant_id, "robot_name": robot_name}
        for subscale_name, cols in subscales.items():
            valid_cols = [c for c in cols if c in group.columns]
            if valid_cols:
                row_data[subscale_name] = group[valid_cols].mean(axis=1).values[0]
            else:
                row_data[subscale_name] = None
        processed_dfs.append(pd.DataFrame([row_data]))

    if not processed_dfs:
        return pd.DataFrame()

    return pd.concat(processed_dfs, ignore_index=True)

clean_scores_df = process_godspeed_scores(scores_df, PARTICIPANT_ID)

if not clean_scores_df.empty:
    subscale_output_filename = f"participant_{PARTICIPANT_ID}_godspeed_subscales.csv"
    clean_scores_df.to_csv(subscale_output_filename, index=False)
    print(f"  [+] Calculated subscales saved to: {subscale_output_filename}")
else:
    print(f"  [!] WARNING: no Godspeed subscales were computed -- {godspeed_file} may be "
          f"empty or missing a 'robot_name' column.")

if "trial_name" in all_trials_timeseries.columns and "robot_name" in all_trials_timeseries.columns:
    all_trials_timeseries = all_trials_timeseries.drop(columns=["trial_name"])

if not clean_scores_df.empty and "participant_id" in all_trials_timeseries.columns and "participant_id" in clean_scores_df.columns:
    individual_master_df = pd.merge(
        all_trials_timeseries,
        clean_scores_df,
        on=["participant_id", "robot_name"],
        how="left"
    )
else:
    individual_master_df = all_trials_timeseries

# ==========================================
# STEP C: SAVE INDIVIDUAL MASTER FILE
# ==========================================
output_filename = f"participant_{PARTICIPANT_ID}_master_integrated.csv"
individual_master_df.to_csv(output_filename, index=False)
print(f"Success! Individual master file saved as: {output_filename}\n")
