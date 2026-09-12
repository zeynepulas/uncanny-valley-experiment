# Human-Robot Interaction (HRI) Data Collection & Analysis Pipeline

This repository contains the complete automated data collection and multimodal analysis pipeline for a thesis experiment investigating human-robot interaction and the uncanny valley. 

The pipeline synchronizes three distinct data streams across multiple robot stimuli:
1. **Subjective Data**: Godspeed questionnaire responses collected via a custom PyQt GUI.
2. **Physiological Data**: Continuous Pulse-to-Pulse Intervals (PPI) and heart rate captured via the Polar Sense SDK.
3. **Behavioral Data**: Webcam video recordings processed natively using **Py-Feat** for automated facial expression and Action Unit (AU) extraction.

---

## Repository Structure

├── MasterScript.py                 # PyQt GUI application for running live experiment trials & data logging
├── IndividualAnalysis.py           # Automated backend pipeline for individual participant processing
├── 

## Pipeline Workflow

### Step 1: Data Collection (`MasterScript.py`)

Run the graphical interface to guide participants through the experiment setup, baseline recording, and 4 distinct robot video trials (`Robot1`, `Robot2`, `Robot3`, `Robot4`).

* Records isolated webcam video streams (`.avi`) per trial.
* Streams real-time physiological data and flags connection drops or invalid sensor contact automatically.
* Collects raw Godspeed questionnaire scores into a structured CSV.

### Step 2: Individual Analysis & Synchronization (`IndividualAnalysis.py`)

Once a participant session is complete, run the analysis script to process their data:

* **Facial Processing:** Downscales video frames and extracts emotional expressions and Action Units at an optimized 1Hz sampling rate via **Py-Feat** (running natively on CPU).
* **Physiological Filtering:** Cleans PPI streams by dropping invalid contact segments and isolating the video playback phase.
* **Millisecond-Level Alignment:** Uses pandas `merge_asof` to time-align high-frequency physiological readings with nearest-neighbor video timestamps.
* **Questionnaire Aggregation:** Computes mean subscale averages for *Anthropomorphism, Animacy, Likeability, Perceived Intelligence, and Perceived Safety*, exporting them to an independent subscale file and merging them row-by-row into the master timeline.

---

## Requirements & Dependencies

* Python 3.11+
* Libraries:
* `pandas`
* `numpy`
* `opencv-python` (`cv2`)
* `Py-Feat` (`feat`)
* External binary: `ffmpeg` (required for video downscaling and stream processing)
