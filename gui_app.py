import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from preprocessing import read_video
from eulerian import fft_filter
from heartrate import find_heart_rate
from stress_analysis import analyze_stress_level
from spo2_analysis import calculate_spo2, get_spo2_color
import os

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, bg='#f0f0f0', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind mouse wheel
        self.scrollable_frame.bind('<Enter>', self._bind_mouse_wheel)
        self.scrollable_frame.bind('<Leave>', self._unbind_mouse_wheel)
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Pack canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Bind canvas resize
        self.canvas.bind('<Configure>', self._on_canvas_configure)
    
    def _on_canvas_configure(self, event):
        # Update the width of the canvas window
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    
    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _bind_mouse_wheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)
    
    def _unbind_mouse_wheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

class HeartRateDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Rate Detection System")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Configure root grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Style configuration
        self.setup_styles()
        
        # Variables
        self.recording = False
        self.previewing = False
        self.video_capture = None
        self.current_frame = None
        self.frames = []
        self.processing = False
        self.video_source = "webcam"
        self.current_video_size = (640, 480)
        self.recording_start_time = None
        self.recording_duration = 60  # seconds
        self.countdown_var = tk.StringVar(value="")
        
        # Create GUI elements
        self.create_widgets()
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)
    
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles with relative font sizes
        default_font_size = self.calculate_font_size(10)
        title_font_size = self.calculate_font_size(24)
        result_font_size = self.calculate_font_size(36)
        
        # Configure progress bar style
        self.style.configure("Processing.Horizontal.TProgressbar",
                           troughcolor='#E0E0E0',
                           background='#2196F3',
                           thickness=20)
        
        self.style.configure('Primary.TButton', 
                           padding=10, 
                           font=('Helvetica', default_font_size, 'bold'),
                           background='#2196F3')
        self.style.configure('Secondary.TButton', 
                           padding=10,
                           font=('Helvetica', default_font_size))
        self.style.configure('Title.TLabel',
                           font=('Helvetica', title_font_size, 'bold'),
                           padding=20,
                           background='#f0f0f0')
        self.style.configure('Result.TLabel',
                           font=('Helvetica', result_font_size, 'bold'),
                           padding=20,
                           foreground='#2196F3',
                           background='#f0f0f0')
        self.style.configure('Status.TLabel',
                           font=('Helvetica', default_font_size),
                           padding=5,
                           background='#f0f0f0')
    
    def calculate_font_size(self, base_size):
        # Calculate font size based on screen resolution
        screen_height = self.root.winfo_screenheight()
        return int(base_size * (screen_height / 1080))  # Scale relative to 1080p
    
    def create_widgets(self):
        # Create scrollable frame
        self.scroll_container = ScrollableFrame(self.root)
        self.scroll_container.grid(row=0, column=0, sticky="nsew")
        
        # Main container
        main_container = ttk.Frame(self.scroll_container.scrollable_frame, padding="20")
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_container, 
                              text="Heart Rate Detection System",
                              style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        # Video display frame
        self.video_frame = ttk.Frame(main_container, padding="10")
        self.video_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Source selection frame
        source_frame = ttk.LabelFrame(main_container, text="Video Source", padding="10")
        source_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        source_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.webcam_btn = ttk.Button(source_frame, 
                                   text="Use Webcam",
                                   style='Primary.TButton',
                                   command=self.use_webcam)
        self.webcam_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.preview_btn = ttk.Button(source_frame,
                                    text="Preview Camera",
                                    style='Secondary.TButton',
                                    command=self.toggle_preview)
        self.preview_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.upload_btn = ttk.Button(source_frame,
                                   text="Upload Video",
                                   style='Secondary.TButton',
                                   command=self.upload_video)
        self.upload_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_container, text="Controls", padding="10")
        controls_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        controls_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.record_button = ttk.Button(controls_frame,
                                      text="Start Recording",
                                      style='Primary.TButton',
                                      command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.process_button = ttk.Button(controls_frame,
                                       text="Process Video",
                                       style='Secondary.TButton',
                                       command=self.process_video,
                                       state=tk.DISABLED)
        self.process_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Add countdown label in the controls frame
        self.countdown_label = ttk.Label(controls_frame,
                                       textvariable=self.countdown_var,
                                       style='Status.TLabel')
        self.countdown_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(main_container, text="Results", padding="10")
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        self.result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_label = ttk.Label(self.result_frame,
                                    text="Heart Rate: -- BPM",
                                    style='Result.TLabel')
        self.result_label.grid(row=0, column=0, sticky="ew")
        
        # Add SpO₂ result label
        self.spo2_label = ttk.Label(self.result_frame,
                                  text="SpO₂: --",
                                  style='Result.TLabel')
        self.spo2_label.grid(row=1, column=0, sticky="ew", pady=10)
        
        # HRV Metrics frame
        self.hrv_frame = ttk.LabelFrame(main_container, text="Heart Rate Variability Metrics", padding="10")
        self.hrv_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        self.hrv_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.sdnn_label = ttk.Label(self.hrv_frame,
                                  text="SDNN: -- ms",
                                  style='Status.TLabel')
        self.sdnn_label.grid(row=0, column=0, sticky="ew", padx=5)
        
        self.rmssd_label = ttk.Label(self.hrv_frame,
                                   text="RMSSD: -- ms",
                                   style='Status.TLabel')
        self.rmssd_label.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Stress Level frame
        self.stress_frame = ttk.LabelFrame(main_container, text="Stress Analysis", padding="10")
        self.stress_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=10)
        self.stress_frame.grid_columnconfigure(0, weight=1)
        
        self.stress_level_label = ttk.Label(self.stress_frame,
                                         text="Stress Level: --",
                                         style='Result.TLabel')
        self.stress_level_label.grid(row=0, column=0, sticky="ew", padx=5)
        
        self.stress_desc_label = ttk.Label(self.stress_frame,
                                        text="",
                                        style='Status.TLabel',
                                        wraplength=400)
        self.stress_desc_label.grid(row=1, column=0, sticky="ew", padx=5)
        
        # Progress frame
        self.progress_frame = ttk.Frame(main_container, padding="10")
        self.progress_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=5)
        self.progress_frame.grid_columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            style="Processing.Horizontal.TProgressbar",
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=5)
        
        # Progress status
        self.progress_status = ttk.Label(
            self.progress_frame,
            text="",
            style='Status.TLabel',
            anchor="center"
        )
        self.progress_status.grid(row=1, column=0, sticky="ew")
        
        # Hide progress frame initially
        self.progress_frame.grid_remove()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_container,
                                  textvariable=self.status_var,
                                  style='Status.TLabel')
        self.status_bar.grid(row=8, column=0, columnspan=2, sticky="ew", pady=5)
    
    def on_window_resize(self, event):
        # Only handle if it's the root window being resized
        if event.widget == self.root:
            # Update video size based on window size
            new_width = min(int(event.width * 0.8), 1280)  # Max width of 1280
            new_height = int(new_width * 0.75)  # 4:3 aspect ratio
            self.current_video_size = (new_width, new_height)
            
            # If there's a current frame, update its display
            if self.current_frame is not None:
                self.update_video_display(self.current_frame)
            
            # Update font sizes
            self.setup_styles()
    
    def update_video_display(self, frame):
        # Always flip horizontally
        frame = cv2.flip(frame, 1)  # 1 for horizontal flip
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.current_video_size)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def use_webcam(self):
        self.video_source = "webcam"
        self.webcam_btn.configure(style='Primary.TButton')
        self.upload_btn.configure(style='Secondary.TButton')
        self.record_button.configure(state=tk.NORMAL)
        self.status_var.set("Webcam selected")
        
    def upload_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        
        if file_path:
            self.video_source = "file"
            self.video_path = file_path
            self.webcam_btn.configure(style='Secondary.TButton')
            self.upload_btn.configure(style='Primary.TButton')
            self.record_button.configure(state=tk.DISABLED)
            self.process_button.configure(state=tk.NORMAL)
            
            # Load video frames
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise Exception("Could not open video file")
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Update status
                self.status_var.set("Loading video frames...")
                self.progress_frame.grid()
                self.progress_bar["value"] = 0
                self.root.update()
                
                # Clear existing frames
                self.frames = []
                
                # Read frames
                frames_read = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Flip frame horizontally for consistency
                    frame = cv2.flip(frame, 1)
                    self.frames.append(frame)
                    
                    # Update progress
                    frames_read += 1
                    progress = (frames_read / total_frames) * 100
                    self.progress_bar["value"] = progress
                    self.status_var.set(f"Loading video: {frames_read}/{total_frames} frames")
                    self.root.update()
                    
                    # Show current frame in display
                    if frames_read == 1:
                        self.current_frame = frame.copy()
                        self.update_video_display(frame)
                
                cap.release()
                
                # Hide progress bar and update status
                self.progress_frame.grid_remove()
                self.status_var.set(f"Video loaded: {os.path.basename(file_path)} ({len(self.frames)} frames)")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading video: {str(e)}")
                self.status_var.set("Error loading video")
                self.progress_frame.grid_remove()
                self.frames = []
                self.process_button.configure(state=tk.DISABLED)
    
    def toggle_preview(self):
        if not self.recording:  # Don't allow preview while recording
            if not self.previewing:
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    messagebox.showerror("Error", "Could not open video capture device")
                    return
                self.previewing = True
                self.preview_btn.configure(text="Stop Preview", style='Primary.TButton')
                self.update_preview()
            else:
                self.previewing = False
                if self.video_capture is not None:
                    self.video_capture.release()
                self.preview_btn.configure(text="Preview Camera", style='Secondary.TButton')
                # Clear the video display
                self.video_label.configure(image='')
    
    def update_preview(self):
        if self.previewing and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame.copy()
                self.update_video_display(frame)
            self.root.after(10, self.update_preview)
    
    def toggle_recording(self):
        if not self.recording and self.video_source == "webcam":
            # Stop preview if it's running
            if self.previewing:
                self.toggle_preview()
            
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open video capture device")
                return
            
            self.recording = True
            self.frames = []
            self.recording_start_time = time.time()
            self.record_button.configure(text="Recording...", state=tk.DISABLED)
            self.process_button.configure(state=tk.DISABLED)
            self.preview_btn.configure(state=tk.DISABLED)
            self.status_var.set("Recording...")
            self.update_video_feed()
            self.update_countdown()
        else:
            self.stop_recording()

    def stop_recording(self):
        self.recording = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.record_button.configure(text="Start Recording", style='Primary.TButton', state=tk.NORMAL)
        self.process_button.configure(text="Process Video", style='Primary.TButton', state=tk.NORMAL)
        self.preview_btn.configure(state=tk.NORMAL)
        self.status_var.set("Recording completed - Ready to process")
        self.countdown_var.set("")

    def update_countdown(self):
        if self.recording:
            elapsed_time = time.time() - self.recording_start_time
            remaining_time = max(0, self.recording_duration - elapsed_time)
            
            if remaining_time > 0:
                self.countdown_var.set(f"Recording: {remaining_time:.1f}s remaining")
                self.root.after(100, self.update_countdown)
            else:
                self.stop_recording()
                self.status_var.set("Recording completed - Ready to process")
    
    def update_video_feed(self):
        if self.recording and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Always flip horizontally
                frame = cv2.flip(frame, 1)
                
                # Store the flipped frame
                self.frames.append(frame)
                
                # Display frame
                self.update_video_display(self.current_frame)
            self.root.after(10, self.update_video_feed)
    
    def process_video(self):
        if not self.frames:
            messagebox.showerror("Error", "No video recorded or loaded")
            return
        
        self.processing = True
        self.process_button.configure(state=tk.DISABLED)
        self.upload_btn.configure(state=tk.DISABLED)
        self.record_button.configure(state=tk.DISABLED)
        
        def process():
            try:
                # Calculate heart rate
                heart_rate = find_heart_rate(self.frames)
                
                # Calculate SpO₂
                spo2, ratio = calculate_spo2(self.frames)
                
                # Calculate stress level
                stress_level = analyze_stress_level(self.frames)
                
                self.root.after(0, lambda: self.update_results(heart_rate, stress_level, spo2))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.processing = False
                self.root.after(0, lambda: self.process_button.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.upload_btn.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.record_button.configure(state=tk.NORMAL))
        
        threading.Thread(target=process, daemon=True).start()
    
    def update_results(self, heart_rate, stress_level, spo2=None):
        if heart_rate is not None:
            self.result_label.configure(text=f"Heart Rate: {heart_rate:.1f} BPM")
        else:
            self.result_label.configure(text="Heart Rate: Failed to detect")
        
        if stress_level is not None:
            self.stress_level_label.configure(text=f"Stress Level: {stress_level}")
        else:
            self.stress_level_label.configure(text="Stress Level: Failed to detect")
            
        if spo2 is not None:
            color = get_spo2_color(spo2)
            self.spo2_label.configure(
                text=f"SpO₂: {spo2:.1f}%",
                foreground=color
            )
        else:
            self.spo2_label.configure(
                text="SpO₂: Failed to detect",
                foreground='#F44336'
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartRateDetectorGUI(root)
    root.mainloop()
