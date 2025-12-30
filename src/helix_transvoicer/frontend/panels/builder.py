"""
Helix Transvoicer - Model Builder Panel.
"""

import queue
import threading
import time
import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List, Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient
from helix_transvoicer.frontend.components.progress import ProgressIndicator
from helix_transvoicer.frontend.components.controls import Slider


class BuilderPanel(ctk.CTkFrame):
    """
    Voice model builder panel.

    Allows users to:
    - Add training samples
    - Configure training parameters
    - Train new voice models
    - Update existing models
    """

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_primary"],
            corner_radius=0,
        )

        self.api_client = api_client
        self._samples: List[Path] = []
        self._model_name: str = ""
        self._training_active = False
        self._training_paused = False
        self._current_job_id = None

        # Thread-safe queue for UI updates (tkinter is NOT thread-safe)
        self._ui_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._start_queue_polling()

    def _build_ui(self):
        """Build the builder panel UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # Title
        title = ctk.CTkLabel(
            self,
            text="MODEL BUILDER",
            font=HelixTheme.FONTS["heading"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 20))

        # Main content
        main_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
        )
        main_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 20))
        main_frame.grid_columnconfigure(0, weight=1)

        # Project section
        self._build_project_section(main_frame)

        # Samples section
        self._build_samples_section(main_frame)

        # Progress section
        self._build_progress_section(main_frame)

        # Side panel
        side_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=300,
        )
        side_frame.grid(row=1, column=1, sticky="ns")

        # Configuration
        self._build_config_section(side_frame)

        # Coverage preview
        self._build_coverage_section(side_frame)

        # Train button
        self.train_btn = ctk.CTkButton(
            side_frame,
            text="▶ START TRAINING",
            height=50,
            **HelixTheme.get_button_style("primary"),
            command=self._on_train,
        )
        self.train_btn.pack(fill="x", padx=20, pady=(20, 10))

        # Training control buttons (hidden by default)
        self.controls_frame = ctk.CTkFrame(side_frame, fg_color="transparent")
        self.controls_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.pause_btn = ctk.CTkButton(
            self.controls_frame,
            text="⏸ Pause",
            height=40,
            **HelixTheme.get_button_style("secondary"),
            command=self._on_pause_resume,
        )
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.cancel_btn = ctk.CTkButton(
            self.controls_frame,
            text="✕ Cancel",
            height=40,
            fg_color=HelixTheme.COLORS["error"],
            hover_color="#b02020",
            text_color="white",
            command=self._on_cancel,
        )
        self.cancel_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Initially hide control buttons
        self.controls_frame.pack_forget()

    def _build_project_section(self, parent):
        """Build project section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        # Title row
        title_row = ctk.CTkFrame(section, fg_color="transparent")
        title_row.pack(fill="x")

        title = ctk.CTkLabel(
            title_row,
            text="PROJECT",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(side="left")

        # Model name input
        name_frame = ctk.CTkFrame(section, fg_color="transparent")
        name_frame.pack(fill="x", pady=10)

        name_label = ctk.CTkLabel(
            name_frame,
            text="Model Name:",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        name_label.pack(anchor="w", pady=(0, 5))

        self.name_entry = ctk.CTkEntry(
            name_frame,
            placeholder_text="Enter model name...",
            width=300,
            height=35,
            **HelixTheme.get_input_style(),
        )
        self.name_entry.pack(anchor="w")

    def _build_samples_section(self, parent):
        """Build training samples section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="both", expand=True, padx=20, pady=20)

        # Title row
        title_row = ctk.CTkFrame(section, fg_color="transparent")
        title_row.pack(fill="x")

        title = ctk.CTkLabel(
            title_row,
            text="TRAINING SAMPLES",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(side="left")

        self.sample_count = ctk.CTkLabel(
            title_row,
            text="0 samples",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.sample_count.pack(side="right")

        # Sample list
        list_frame = ctk.CTkFrame(
            section,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            corner_radius=HelixTheme.RADIUS["sm"],
        )
        list_frame.pack(fill="both", expand=True, pady=10)

        self.sample_list = ctk.CTkTextbox(
            list_frame,
            fg_color="transparent",
            text_color=HelixTheme.COLORS["text_primary"],
            font=HelixTheme.FONTS["mono"],
            state="disabled",
        )
        self.sample_list.pack(fill="both", expand=True, padx=10, pady=10)

        # Buttons
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x")

        add_btn = ctk.CTkButton(
            btn_frame,
            text="+ Add Files",
            width=120,
            **HelixTheme.get_button_style("secondary"),
            command=self._on_add_samples,
        )
        add_btn.pack(side="left", padx=(0, 10))

        remove_btn = ctk.CTkButton(
            btn_frame,
            text="🗑 Remove All",
            width=120,
            **HelixTheme.get_button_style("ghost"),
            command=self._on_clear_samples,
        )
        remove_btn.pack(side="left")

    def _build_progress_section(self, parent):
        """Build training progress section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="TRAINING PROGRESS",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        self.progress = ProgressIndicator(section, width=600)
        self.progress.pack(fill="x")

    def _build_config_section(self, parent):
        """Build configuration section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="TRAINING CONFIGURATION",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 15))

        # Epochs slider
        self.epochs_slider = Slider(
            section,
            label="Epochs:",
            from_=10,
            to=500,
            default=100,
            format_str="{:.0f}",
            width=260,
        )
        self.epochs_slider.pack(fill="x", pady=5)

        # Batch size slider
        self.batch_slider = Slider(
            section,
            label="Batch Size:",
            from_=1,
            to=64,
            default=16,
            format_str="{:.0f}",
            width=260,
        )
        self.batch_slider.pack(fill="x", pady=5)

        # Learning rate slider
        self.lr_slider = Slider(
            section,
            label="Learn Rate:",
            from_=0.00001,
            to=0.001,
            default=0.0001,
            format_str="{:.5f}",
            width=260,
        )
        self.lr_slider.pack(fill="x", pady=5)

        # Options
        options = ctk.CTkFrame(section, fg_color="transparent")
        options.pack(fill="x", pady=10)

        self.denoise_var = ctk.BooleanVar(value=True)
        denoise_cb = ctk.CTkCheckBox(
            options,
            text="Auto-denoise samples",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.denoise_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        denoise_cb.pack(anchor="w", pady=2)

        self.augment_var = ctk.BooleanVar(value=True)
        augment_cb = ctk.CTkCheckBox(
            options,
            text="Augment training data",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.augment_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        augment_cb.pack(anchor="w", pady=2)

        self.silence_var = ctk.BooleanVar(value=True)
        silence_cb = ctk.CTkCheckBox(
            options,
            text="Remove silence from audio",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.silence_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        silence_cb.pack(anchor="w", pady=2)

        self.autosplit_var = ctk.BooleanVar(value=True)
        autosplit_cb = ctk.CTkCheckBox(
            options,
            text="Auto-split long audio",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.autosplit_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        autosplit_cb.pack(anchor="w", pady=2)

    def _build_coverage_section(self, parent):
        """Build emotion coverage preview."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="EMOTION COVERAGE",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Coverage bars (simplified)
        emotions = ["Neutral", "Happy", "Sad", "Angry", "Calm"]
        self.coverage_bars = {}

        for emotion in emotions:
            row = ctk.CTkFrame(section, fg_color="transparent")
            row.pack(fill="x", pady=2)

            label = ctk.CTkLabel(
                row,
                text=f"{emotion}:",
                font=HelixTheme.FONTS["small"],
                text_color=HelixTheme.COLORS["text_tertiary"],
                width=60,
                anchor="w",
            )
            label.pack(side="left")

            bar = ctk.CTkProgressBar(
                row,
                width=140,
                height=8,
                fg_color=HelixTheme.COLORS["bg_tertiary"],
                progress_color=HelixTheme.COLORS["accent_dim"],
            )
            bar.set(0)
            bar.pack(side="left", padx=5)

            pct = ctk.CTkLabel(
                row,
                text="0%",
                font=HelixTheme.FONTS["small"],
                text_color=HelixTheme.COLORS["text_tertiary"],
                width=35,
            )
            pct.pack(side="left")

            self.coverage_bars[emotion.lower()] = (bar, pct)

    def _update_sample_list(self):
        """Update sample list display."""
        self.sample_list.configure(state="normal")
        self.sample_list.delete("1.0", "end")

        for i, path in enumerate(self._samples, 1):
            self.sample_list.insert("end", f"{i}. {path.name}\n")

        self.sample_list.configure(state="disabled")
        self.sample_count.configure(text=f"{len(self._samples)} samples")

    def _on_add_samples(self):
        """Add sample files."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac"),
            ("All files", "*.*"),
        ]

        paths = filedialog.askopenfilenames(
            title="Select Training Samples",
            filetypes=filetypes,
        )

        if paths:
            for path in paths:
                p = Path(path)
                if p not in self._samples:
                    self._samples.append(p)
            self._update_sample_list()

    def _on_clear_samples(self):
        """Clear all samples."""
        self._samples.clear()
        self._update_sample_list()

    def _start_queue_polling(self):
        """Start polling the UI update queue (runs on main thread)."""
        self._process_ui_queue()

    def _process_ui_queue(self):
        """Process pending UI updates from background threads."""
        try:
            # Process all pending updates (non-blocking)
            while True:
                callback, args, kwargs = self._ui_queue.get_nowait()
                callback(*args, **kwargs)
        except queue.Empty:
            pass
        # Schedule next check (keeps GUI responsive)
        self.after(50, self._process_ui_queue)

    def _queue_update(self, callback, *args, **kwargs):
        """Queue a UI update from a background thread (thread-safe)."""
        self._ui_queue.put((callback, args, kwargs))

    def _on_train(self):
        """Start training."""
        # Read ALL GUI values on main thread BEFORE starting background thread
        # (tkinter widgets are NOT thread-safe)
        model_name = self.name_entry.get().strip()
        if not model_name:
            self.progress.set_error("Enter a model name")
            return

        if len(self._samples) < 1:
            self.progress.set_error("Add at least 1 audio sample")
            return

        # Capture all values needed by background thread
        samples_copy = list(self._samples)  # Make a copy
        epochs = int(self.epochs_slider.get())
        batch_size = int(self.batch_slider.get())
        remove_silence = self.silence_var.get()
        auto_split = self.autosplit_var.get()

        self.progress.reset()
        self.progress.set_stage("Uploading samples...")
        self.train_btn.configure(state="disabled")
        self._current_job_id = None
        self._training_active = True
        self._training_paused = False

        # Show control buttons
        self.pause_btn.configure(text="⏸ Pause")
        self.controls_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Single background thread handles upload + polling
        def training_worker():
            try:
                # Step 1: Upload and submit job (use pre-captured values)
                result = self.api_client.train_model(
                    model_name,
                    samples_copy,
                    epochs=epochs,
                    batch_size=batch_size,
                    remove_silence=remove_silence,
                    auto_split=auto_split,
                )
                job_id = result.get("job_id")
                if not job_id:
                    self._queue_update(self._on_train_error, "No job ID returned")
                    return

                self._current_job_id = job_id
                self._queue_update(self.progress.set_stage, "Training...")

                # Step 2: Poll for progress in same thread
                while self._training_active:
                    time.sleep(1)  # Poll every second

                    try:
                        job = self.api_client.get_job(job_id)
                        status = job.get("status", "unknown")
                        progress = job.get("progress", 0)
                        stage = job.get("stage", "")

                        if status == "completed":
                            self._queue_update(self._on_train_complete, job)
                            return
                        elif status == "failed":
                            self._queue_update(self._on_train_error, stage or "Training failed")
                            return
                        elif status == "cancelled":
                            self._queue_update(self._on_train_error, "Training cancelled")
                            return
                        else:
                            # Update progress via queue (thread-safe)
                            self._queue_update(self._update_progress, progress, stage)
                    except Exception as e:
                        self._queue_update(self._on_train_error, str(e))
                        return

            except Exception as e:
                self._queue_update(self._on_train_error, str(e))

        thread = threading.Thread(target=training_worker, daemon=True)
        thread.start()

    def _update_progress(self, progress: float, stage: str):
        """Update progress indicator."""
        self.progress.set_progress(progress)
        if stage:
            self.progress.set_stage(stage)

    def _on_train_complete(self, result):
        """Handle training completion (called from main thread)."""
        self._training_active = False
        self._training_paused = False
        self.progress.set_complete()
        self.train_btn.configure(state="normal")
        self.controls_frame.pack_forget()
        self._current_job_id = None

    def _on_train_error(self, error_msg: str):
        """Handle training error (called from main thread)."""
        self._training_active = False
        self._training_paused = False
        self.progress.set_error(error_msg[:50])
        self.train_btn.configure(state="normal")
        self.controls_frame.pack_forget()
        self._current_job_id = None

    def _on_pause_resume(self):
        """Toggle pause/resume training."""
        if not self._current_job_id:
            return

        if self._training_paused:
            # Resume
            try:
                self.api_client.resume_job(self._current_job_id)
                self._training_paused = False
                self.pause_btn.configure(text="⏸ Pause")
                self.progress.set_stage("Resuming...")
            except Exception as e:
                self.progress.set_error(f"Resume failed: {str(e)[:30]}")
        else:
            # Pause
            try:
                self.api_client.pause_job(self._current_job_id)
                self._training_paused = True
                self.pause_btn.configure(text="▶ Resume")
                self.progress.set_stage("Paused")
            except Exception as e:
                self.progress.set_error(f"Pause failed: {str(e)[:30]}")

    def _on_cancel(self):
        """Cancel training with confirmation."""
        if not self._current_job_id:
            return

        # Show confirmation dialog
        result = messagebox.askyesno(
            "Cancel Training",
            "Are you sure you want to cancel training?\n\n"
            "This will delete the partially trained model.",
            icon="warning",
        )

        if not result:
            return

        model_name = self.name_entry.get().strip()

        try:
            # Cancel the job
            self.api_client.cancel_job(self._current_job_id)

            # Delete the partial model
            if model_name:
                try:
                    self.api_client.delete_model(model_name)
                except Exception:
                    pass  # Model might not exist yet

            self._training_active = False
            self._training_paused = False
            self.progress.set_error("Training cancelled")
            self.train_btn.configure(state="normal")
            self.controls_frame.pack_forget()
            self._current_job_id = None

        except Exception as e:
            self.progress.set_error(f"Cancel failed: {str(e)[:30]}")
