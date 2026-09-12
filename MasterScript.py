import time
import asyncio
import cv2
import random
import os
import threading
import sys
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox
import pygame
from bleak import BleakScanner
from polar_python import PolarDevice

# --- EXPERIMENT CONFIGURATION ---
ROBOT_TRIALS = [
    {"name": "Robot1", "display_name": "Mechanical / Industrial", "video": "robot1.mp4"},
    {"name": "Robot2", "display_name": "Social / Friendly (Stylized)", "video": "robot2.mp4"},
    {"name": "Robot3", "display_name": "Humanoid", "video": "robot3.mp4"},
    {"name": "Robot4", "display_name": "Super Realistic / Eerie", "video": "robot4.mp4"},
]

GODSPEED_DIMENSIONS = {
    "Anthropomorphism": [
        ("Fake / Gekünstelt", "Natural / Natürlich", "anthropomorphism_1"),
        ("Machinelike / Maschinenartig", "Humanlike / Menschenähnlich", "anthropomorphism_2"),
        ("Unconscious / Unbewusst", "Conscious / Bewusst", "anthropomorphism_3"),
        ("Artificial / Künstlich", "Lifelike / Lebensecht", "anthropomorphism_4"),
        ("Moving rigidly / Starr bewegend", "Moving elegantly / Elegant bewegend", "anthropomorphism_5")
    ],
    "Animacy": [
        ("Dead / Tot", "Alive / Lebendig", "animacy_1"),
        ("Stagnant / Unbewegt", "Lively / Lebhaft", "animacy_2"),
        ("Mechanical / Mechanisch", "Organic / Organisch", "animacy_3"),
        ("Artificial / Künstlich", "Lifelike / Lebensecht", "animacy_4"),
        ("Inert / Träge", "Interactive / Interaktiv", "animacy_5"),
        ("Apathetic / Apathisch", "Responsive / Reaktionsfähig", "animacy_6")
    ],
    "Likeability": [
        ("Dislike / Unsympathisch", "Like / Sympathisch", "likeability_1"),
        ("Unfriendly / Unfreundlich", "Friendly / Freundlich", "likeability_2"),
        ("Unkind / Ungütig", "Kind / Gütig", "likeability_3"),
        ("Unpleasant / Unangenehm", "Pleasant / Angenehm", "likeability_4"),
        ("Awful / Schrecklich", "Nice / Schön", "likeability_5")
    ],
    "Perceived Intelligence": [
        ("Incompetent / Inkompetent", "Competent / Kompetent", "intelligence_1"),
        ("Ignorant / Unwissend", "Knowledgeable / Sachkundig", "intelligence_2"),
        ("Irresponsible / Unverantwortlich", "Responsible / Verantwortlich", "intelligence_3"),
        ("Unintelligent / Unintelligent", "Intelligent / Intelligent", "intelligence_4"),
        ("Foolish / Töricht", "Sensible / Vernünftig", "intelligence_5")
    ],
    "Perceived Safety": [
        ("Anxious / Ängstlich", "Relaxed / Entspannt", "safety_1"),
        ("Agitated / Aufgeregt", "Calm / Ruhig", "safety_2"),
        ("Quiescent / Ruhig", "Surprised / Überrascht", "safety_3")
    ]
}

class UncannyValleyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uncanny Valley Experiment")
        self.root.attributes('-fullscreen', True)  
        self.root.configure(bg="#f4f6f9")  

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
        except Exception as e:
            print(f"[!] Warning: Could not initialize pygame mixer: {e}")

        self.participant_id = simpledialog.askstring("Participant ID", "Enter Participant Number (e.g., 1, 2, 3):", parent=root)
        if not self.participant_id:
            self.participant_id = "1"  

        self.current_trial_idx = 0
        self.is_experiment_running = False
        self.is_trial_recording = False
        self.ppi_file = None  # Safe initialization
        self.cap = None  # Webcam handle -- opened once for the whole session, see initialize_camera()
        
        self.current_trial_name = "Preparation"
        self.current_phase = "Setup"
        self.trial_start_time = time.time()

        self.container = tk.Frame(root, bg="#f4f6f9")
        self.container.pack(fill="both", expand=True)

        self.show_welcome_screen()

    def clear_screen(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_welcome_screen(self):
        self.clear_screen()

        title = tk.Label(self.container, text=f"Uncanny Valley Study", font=("Arial", 28, "bold"), fg="#2c3e50", bg="#f4f6f9")
        title.pack(pady=40)

        instructions = (
            "Welcome to the experiment.\n\n"
            f"You will be presented with a series of {len(ROBOT_TRIALS)} robot stimulus videos.\n\n"
            "- It will take approximately 10 minutes.\n\n"
            "- Before each video, there will be a 30-second resting baseline.\n\n"
            "- After each video, you will complete the questionnaire on the screen.\n\n"
            "- After the completion of each questionnaire the next video will play automatically.\n\n"
            "- Please remain still, relaxed, and focused on the screen.\n\n"
            "- Please try not to move the arm that the sensor is attached to.\n\n\n\n"
        )
        lbl = tk.Label(self.container, text=instructions, font=("Arial", 15), fg="#4f5b66", bg="#f4f6f9", justify="center")
        lbl.pack(pady=15)

        self.exp_frame = tk.Frame(self.container, bg="#eef2f7", bd=1, relief="solid", padx=20, pady=15)
        self.exp_frame.pack(pady=20)

        exp_title = tk.Label(self.exp_frame, text="[Control Panel]", font=("Arial", 11, "bold"), fg="#e67e22", bg="#eef2f7")
        exp_title.pack(pady=(0, 5))

        self.connect_btn = tk.Button(
            self.exp_frame, 
            text="1. Connect Bluetooth Sensor (Polar Sense)", 
            font=("Arial", 12, "bold"), 
            bg="#f39c12", 
            fg="black", 
            padx=15, 
            pady=8, 
            bd=0, 
            cursor="hand2", 
            command=self.connect_bluetooth_sensor
        )
        self.connect_btn.pack(pady=5)

        self.connection_status_lbl = tk.Label(self.exp_frame, text="Status: Not Connected (Optional for Testing)", font=("Arial", 11, "italic"), fg="#7f8c8d", bg="#eef2f7")
        self.connection_status_lbl.pack(pady=5)

        # Enabled by default so you can test without connecting hardware!
        self.start_btn = tk.Button(
            self.container, 
            text="Start Experiment (Test Mode Enabled)", 
            font=("Arial", 14, "bold"), 
            bg="#2ecc71", 
            fg="black", 
            padx=25, 
            pady=12, 
            bd=0, 
            state="normal", 
            cursor="hand2", 
            command=self.begin_experiment_session
        )
        self.start_btn.pack(pady=20)

    def connect_bluetooth_sensor(self):
        self.connect_btn.config(state="disabled", text="Connecting...")
        self.connection_status_lbl.config(text="Status: Scanning for Polar Sense...", fg="#d35400")

        ppi_filename = f"participant_{self.participant_id}_continuous_ppi.csv"
        try:
            self.ppi_file = open(ppi_filename, "w", buffering=1)
            self.ppi_file.write("absolute_time,elapsed_trial_sec,trial_name,phase,ppi_ms,heart_rate,error_estimate,invalid_ppi,skin_contact_status\n")
        except Exception as e:
            messagebox.showerror("File Error", f"Could not create CSV file: {e}")
            self.connect_btn.config(state="normal", text="1. Connect Bluetooth Sensor (Polar Sense)")
            self.connection_status_lbl.config(text="Status: File Error", fg="#c0392b")
            return

        self.is_experiment_running = True
        self.experiment_start_time = time.time()

        self.polar_thread = threading.Thread(target=self.run_continuous_polar_stream, daemon=True)
        self.polar_thread.start()

    def run_continuous_polar_stream(self):
        async def _stream():
            try:
                print("[*] Scanning for Polar Sense...")
                device = await BleakScanner.find_device_by_filter(
                    lambda bd, ad: bd.name and "Polar Sense" in bd.name, 
                    timeout=15.0
                )
                if not device:
                    print("[!] Polar Sense device not found.")
                    self.root.after(0, lambda: self.connection_failed_ui("Polar Sense device not found during scan."))
                    return

                print(f"[*] Connected to {device.name}. Starting continuous stream...")
                async with PolarDevice(device) as polar:
                    def ppi_callback(data):
                        if not self.is_experiment_running or not self.ppi_file:
                            return
                        
                        now = time.time()
                        elapsed_trial_sec = now - self.trial_start_time
                        samples = getattr(data, 'samples', [])
                        
                        for sample in samples:
                            ppi_val = getattr(sample, 'ppi', 0)
                            heart_rate = getattr(sample, 'hr', 0)
                            error_est = getattr(sample, 'error_estimate', 0)
                            invalid = getattr(sample, 'invalid_ppi', False)
                            skin_contact = getattr(sample, 'skin_contact_status', True)

                            self.ppi_file.write(f"{now:.3f},{elapsed_trial_sec:.3f},{self.current_trial_name},{self.current_phase},{ppi_val},{heart_rate},{error_est},{invalid},{skin_contact}\n")
                            self.ppi_file.flush()

                    await polar.start_ppi_stream(ppi_callback=ppi_callback)
                    self.root.after(0, self.connection_success_ui)

                    while self.is_experiment_running:
                        await asyncio.sleep(0.5)

            except Exception:
                print("[!] Error in continuous Polar stream: disconnected")
                self.root.after(0, lambda: self.connection_failed_ui("Disconnected"))
            
        asyncio.run(_stream())

    def connection_success_ui(self):
        self.connection_status_lbl.config(text="Status: Connected & Streaming ✔", fg="#27ae60")
        self.connect_btn.config(text="Connected Successfully", bg="#2ecc71")
        messagebox.showinfo("Bluetooth Ready", "Polar Sense connected successfully!")

    def connection_failed_ui(self, err_msg):
        self.is_experiment_running = False
        self.connect_btn.config(state="normal", text="1. Connect Bluetooth Sensor (Polar Sense)", bg="#f39c12")
        self.connection_status_lbl.config(text="Status: Connection Failed (Test mode available)", fg="#c0392b")

    def begin_experiment_session(self):
        if not self.ppi_file:
            ppi_filename = f"participant_{self.participant_id}_continuous_ppi.csv"
            try:
                self.ppi_file = open(ppi_filename, "w", buffering=1)
                self.ppi_file.write("absolute_time,elapsed_trial_sec,trial_name,phase,ppi_ms,heart_rate,error_estimate,invalid_ppi,skin_contact_status\n")
            except Exception as e:
                print(f"[!] Warning: Could not create test PPI file: {e}")

        if self.cap is None:
            camera_ok = self.initialize_camera()
            if not camera_ok:
                messagebox.showerror(
                    "Camera Error",
                    "Could not get a working webcam stream after several attempts.\n\n"
                    "Check that no other app (Zoom, Photo Booth, another Python process, "
                    "etc.) is using the camera, and that an iPhone isn't nearby acting as "
                    "a Continuity Camera, then try again."
                )
                return

        self.is_experiment_running = True
        self.experiment_start_time = time.time()
        self.start_next_trial()

    def initialize_camera(self, max_attempts=5, retry_delay=1.5, warmup_frames=10):
        """Open the webcam ONCE for the whole experiment and confirm it's actually
        delivering frames before proceeding -- opening/closing a fresh
        cv2.VideoCapture every trial is a known-flaky pattern on macOS/AVFoundation
        (isOpened() can return True while .read() keeps failing), which is what was
        producing trials with 0 frames written. This keeps a single capture session
        alive across all trials instead of recreating it each time."""
        for attempt in range(1, max_attempts + 1):
            print(f"[*] Opening webcam (attempt {attempt}/{max_attempts})...")
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30.0)

            if not cap.isOpened():
                print(f"[!] Attempt {attempt}: camera did not open.")
                cap.release()
                time.sleep(retry_delay)
                continue

            # Warm-up: isOpened() can be True while .read() keeps failing, so confirm
            # real frames are actually coming through before trusting this session.
            good_frames = 0
            for _ in range(warmup_frames):
                ret, frame = cap.read()
                if ret and frame is not None:
                    good_frames += 1
                time.sleep(0.03)

            if good_frames >= warmup_frames // 2:
                print(f"[+] Camera confirmed working ({good_frames}/{warmup_frames} warm-up frames captured).")
                self.cap = cap
                return True
            else:
                print(f"[!] Attempt {attempt}: camera opened but only {good_frames}/{warmup_frames} "
                      f"warm-up frames succeeded -- treating as a bad session and retrying.")
                cap.release()
                time.sleep(retry_delay)

        print("[!] ERROR: could not get a working camera stream after multiple attempts.")
        self.cap = None
        return False

    def start_next_trial(self):
        if self.current_trial_idx < len(ROBOT_TRIALS):
            trial_data = ROBOT_TRIALS[self.current_trial_idx]
            self.current_trial_name = trial_data["name"]
            self.current_phase = "baseline"
            self.trial_start_time = time.time()
            self.show_baseline_screen(trial_data)
        else:
            self.show_completion_screen()

    def show_baseline_screen(self, trial_data):
        self.clear_screen()

        header = tk.Label(self.container, text=f"Trial {self.current_trial_idx + 1} of {len(ROBOT_TRIALS)}", font=("Arial", 18), fg="#7f8c8d", bg="#f4f6f9")
        header.pack(pady=40)

        msg = tk.Label(self.container, text="Resting Baseline (30 seconds)\n\nPlease look at the screen and relax.\n The next video will play automatically after the countdown.", font=("Arial", 22, "bold"), fg="#2c3e50", bg="#f4f6f9", justify="center")
        msg.pack(pady=40)

        self.timer_lbl = tk.Label(self.container, text="30", font=("Arial", 40, "bold"), fg="#27ae60", bg="#f4f6f9")
        self.timer_lbl.pack(pady=20)

        self.baseline_seconds_left = 30
        self.run_baseline_countdown(trial_data)

    def run_baseline_countdown(self, trial_data):
        if self.baseline_seconds_left > 0:
            self.timer_lbl.config(text=str(self.baseline_seconds_left))
            self.baseline_seconds_left -= 1
            self.root.after(1000, lambda: self.run_baseline_countdown(trial_data))
        else:
            self.show_video_playback_screen(trial_data)

    def show_video_playback_screen(self, trial_data):
        self.clear_screen()

        robot_name = trial_data["name"]
        video_path = trial_data["video"]

        self.current_trial_name = robot_name
        self.current_phase = "video_playback"
        self.trial_start_time = time.time()

        self.root.withdraw()

        # Start independent webcam recording thread for this specific trial
        self.is_trial_recording = True
        recording_thread = threading.Thread(target=self.record_webcam_isolated, args=(robot_name,))
        recording_thread.start()

        if not os.path.exists(video_path):
            print(f"[!] Error: Video file {video_path} not found.")
            self.is_trial_recording = False
            recording_thread.join()
            self.root.deiconify()
            self.show_questionnaire_screen(trial_data)
            return

        # Launch VLC natively via subprocess for absolute hardware-level A/V sync & full screen
        try:
            vlc_path = "/Applications/VLC.app/Contents/MacOS/VLC"
            if os.path.exists(vlc_path):
                subprocess.run([vlc_path, "--fullscreen", "--play-and-exit", video_path])
            else:
                messagebox.showerror("VLC Error", "VLC application not found in /Applications. Please install VLC.")
        except Exception as e:
            print(f"[!] Error running VLC process: {e}")

        self.is_trial_recording = False
        recording_thread.join()

        self.root.deiconify()
        self.show_questionnaire_screen(trial_data)

    def record_webcam_isolated(self, robot_name):
        # The camera is opened ONCE for the whole session (see initialize_camera,
        # called from begin_experiment_session) rather than per trial -- repeatedly
        # opening/releasing cv2.VideoCapture on macOS/AVFoundation is what was
        # causing every other trial to silently record 0 frames.
        if self.cap is None or not self.cap.isOpened():
            print(f"[!] Warning: camera is not open for trial {robot_name} -- attempting to recover...")
            if not self.initialize_camera():
                print(f"[!] Error: could not recover the camera for trial {robot_name}. "
                      f"No video will be recorded for this trial.")
                return

        # Log the true wall-clock instant this trial's recording actually starts, so
        # IndividualAnalysis.py can measure the real offset between this and
        # self.trial_start_time (set when the video_playback phase began) instead of
        # assuming they're the same moment. Note the camera itself may have opened
        # well before this (at experiment start), so this offset is now generally
        # small and stable rather than the ~1.5s+ gap from opening fresh each trial.
        camera_start_time = time.time()
        timing_log_filename = f"participant_{self.participant_id}_camera_timing.csv"
        log_exists = os.path.exists(timing_log_filename)
        try:
            with open(timing_log_filename, "a") as f:
                if not log_exists:
                    f.write("robot_name,camera_start_time\n")
                f.write(f"{robot_name},{camera_start_time:.3f}\n")
        except Exception as e:
            print(f"[!] Warning: could not write camera timing log: {e}")

        video_filename = f"participant_{self.participant_id}_{robot_name}_cam.avi"
        fourcc = cv2.VideoWriter_fourcc(*'M','J','P','G')
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

        frames_written = 0
        consecutive_failed_reads = 0
        warned_mid_trial = False
        while self.is_trial_recording:
            ret_cam, frame_cam = self.cap.read()
            if ret_cam and frame_cam is not None:
                frame_cam_resized = cv2.resize(frame_cam, (640, 480))
                out.write(frame_cam_resized)
                frames_written += 1
                consecutive_failed_reads = 0
            else:
                consecutive_failed_reads += 1
                time.sleep(0.01)
                # ~3 seconds of back-to-back failed reads means the stream likely died
                # mid-trial -- flag it immediately instead of only finding out afterward.
                if consecutive_failed_reads == 300 and not warned_mid_trial:
                    print(f"[!] WARNING: camera stream appears to have stopped delivering "
                          f"frames partway through {robot_name} (300+ consecutive failed reads).")
                    warned_mid_trial = True

        # Note: self.cap is intentionally NOT released here -- it stays open for the
        # next trial and is only released at the very end (show_completion_screen).
        out.release()
        print(f"[*] Trial {robot_name} webcam saved: {frames_written} frames written.")
        if frames_written == 0:
            print(f"[!] CRITICAL: 0 frames written for {robot_name} -- this trial has NO facial video data.")

    def show_questionnaire_screen(self, trial_data):
        self.clear_screen()
        robot_name = trial_data["name"]
        display_name = trial_data["display_name"]

        self.current_trial_name = robot_name
        self.current_phase = "questionnaire"
        self.trial_start_time = time.time()

        main_frame = tk.Frame(self.container, bg="#f4f6f9")
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, bg="#f4f6f9", highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f4f6f9")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(80, 0))
        scrollbar.pack(side="right", fill="y", padx=(0, 80))

        title = tk.Label(scrollable_frame, text=f"Questionnaire: {display_name}", font=("Arial", 22, "bold"), fg="#2c3e50", bg="#f4f6f9")
        title.pack(pady=(20, 5))

        instruction = tk.Label(scrollable_frame, text="Please rate your impression on the following scales (1 = Low to 5 = High).\nPlease scroll down using the scrollbar on the right side of the screen and click 'Submit' when finished.", font=("Arial", 12), fg="#7f8c8d", bg="#f4f6f9")
        instruction.pack(pady=(0, 20))

        self.rating_vars = {}

        for category, items in GODSPEED_DIMENSIONS.items():
            cat_label = tk.Label(scrollable_frame, text=f"{category.upper()}", font=("Arial", 13, "bold"), fg="black", bg="#f4f6f9", anchor="center")
            cat_label.pack(anchor="center", pady=(15, 5))

            card_frame = tk.Frame(scrollable_frame, bg="white", bd=1, relief="solid")
            card_frame.pack(padx=20, pady=5, fill="x")

            for left_lbl_text, right_lbl_text, key in items:
                row_frame = tk.Frame(card_frame, bg="white")
                row_frame.pack(fill="x", padx=25, pady=10)

                left_lbl = tk.Label(row_frame, text=left_lbl_text, font=("Arial", 10, "bold"), fg="#34495e", bg="white", anchor="w", width=35)
                left_lbl.pack(side="left", padx=(5, 10))

                var = tk.IntVar(value=3)
                self.rating_vars[key] = var

                radio_container = tk.Frame(row_frame, bg="white")
                radio_container.pack(side="left", expand=True)

                for val in range(1, 6):
                    rb = tk.Radiobutton(
                        radio_container, 
                        text=str(val), 
                        variable=var, 
                        value=val, 
                        font=("Arial", 10, "bold"), 
                        bg="white", 
                        fg="#2c3e50", 
                        selectcolor="#ecf0f1",
                        takefocus=False,
                        cursor="hand2"
                    )
                    rb.pack(side="left", padx=15)

                right_lbl = tk.Label(row_frame, text=right_lbl_text, font=("Arial", 10, "bold"), fg="#34495e", bg="white", anchor="e", width=35)
                right_lbl.pack(side="right", padx=(10, 5))

                def make_clickable(widget, target_var, target_val):
                    widget.bind("<Button-1>", lambda e: target_var.set(target_val))

                make_clickable(row_frame, var, val)
                make_clickable(left_lbl, var, val)
                make_clickable(right_lbl, var, val)

        submit_btn = tk.Button(scrollable_frame, text="Submit Ratings", font=("Arial", 14, "bold"), bg="#3498db", fg="black", padx=30, pady=12, bd=0, cursor="hand2", command=lambda: self.save_questionnaire(robot_name))
        submit_btn.pack(pady=30)

    def save_questionnaire(self, robot_name):
        scores = {key: var.get() for key, var in self.rating_vars.items()}

        score_filename = f"participant_{self.participant_id}_godspeed_scores.csv"
        file_exists = os.path.exists(score_filename)

        all_keys = [item[2] for cat in GODSPEED_DIMENSIONS.values() for item in cat]

        with open(score_filename, "a") as f:
            if not file_exists:
                header = "participant_id,robot_name," + ",".join(all_keys) + "\n"
                f.write(header)
            
            row_data = f"{self.participant_id},{robot_name}," + ",".join(str(scores[k]) for k in all_keys) + "\n"
            f.write(row_data)

        self.current_trial_idx += 1
        self.start_next_trial()

    def show_completion_screen(self, robot_name=None):
        self.clear_screen()
        self.is_experiment_running = False
        self.current_phase = "completed"
        self.current_trial_name = "None"

        if hasattr(self, 'ppi_file') and self.ppi_file:
            try:
                self.ppi_file.close()
            except Exception:
                pass

        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        center_wrapper = tk.Frame(self.container, bg="#f4f6f9")
        center_wrapper.place(relx=0.5, rely=0.5, anchor="center")

        title = tk.Label(center_wrapper, text="Experiment Complete!", font=("Arial", 28, "bold"), fg="#27ae60", bg="#f4f6f9")
        title.pack(pady=20)

        msg = tk.Label(center_wrapper, text=f"You have finished successfully.\nThank you for your patience :)", font=("Arial", 16), fg="#34495e", bg="#f4f6f9", justify="center")
        msg.pack(pady=20)

        exit_btn = tk.Button(center_wrapper, text="Exit", font=("Arial", 14, "bold"), bg="#e74c3c", fg="black", padx=20, pady=10, bd=0, cursor="hand2", command=self.root.destroy)
        exit_btn.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = UncannyValleyApp(root)
    root.mainloop()