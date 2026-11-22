"""WorkShield desktop application for facial-image assessment."""
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
from Python.config import MODEL_PATH, EVALUATION_JSON_PATH
from Python.inference import load_workshield_model, predict_loaded_model

class WorkShieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WorkShield | AI Facial Image Assessment")
        self.root.geometry("1120x780")
        self.root.minsize(940, 680)
        self.model = None
        self.metadata = None
        self.selected_image = None
        self.preview_reference = None
        self.model_status = tk.StringVar(value="Checking for a trained model...")
        self._configure_style()
        self._build_interface()
        threading.Thread(target=self._load_model_background, daemon=True).start()

    def _configure_style(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=9)

    def _build_interface(self):
        header = ttk.Frame(self.root, padding=(22, 18))
        header.pack(fill="x")
        ttk.Label(header, text="WORKSHIELD", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="AI-Powered Remote Identity and Workstation Verification", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 8))
        ttk.Label(header, textvariable=self.model_status).pack(anchor="w")
        ttk.Separator(self.root).pack(fill="x", padx=20, pady=(0, 10))
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.assessment_tab = ttk.Frame(self.notebook, padding=18)
        self.analytics_tab = ttk.Frame(self.notebook, padding=18)
        self.about_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.assessment_tab, text="Image Assessment")
        self.notebook.add(self.analytics_tab, text="Model Analytics")
        self.notebook.add(self.about_tab, text="About & Limitations")
        self._build_assessment_tab()
        self._build_analytics_tab()
        self._build_about_tab()

    def _build_assessment_tab(self):
        self.assessment_tab.columnconfigure(0, weight=1)
        self.assessment_tab.columnconfigure(1, weight=1)
        self.assessment_tab.rowconfigure(1, weight=1)
        ttk.Label(self.assessment_tab, text="Facial Image Assessment", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        image_frame = ttk.LabelFrame(self.assessment_tab, text="Selected Image", padding=12)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.preview_label = ttk.Label(image_frame, text="No image selected\n\nChoose a facial image to begin.", anchor="center", justify="center")
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)
        image_buttons = ttk.Frame(image_frame)
        image_buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(image_buttons, text="Select Image", style="Action.TButton", command=self.select_image).pack(side="left", padx=(0, 8))
        ttk.Button(image_buttons, text="Clear", command=self.clear_image).pack(side="left")

        result_frame = ttk.LabelFrame(self.assessment_tab, text="Assessment Results", padding=14)
        result_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        self.result_class = tk.StringVar(value="Awaiting image analysis")
        self.fake_score = tk.StringVar(value="FAKE model score: --")
        self.real_score = tk.StringVar(value="REAL model score: --")
        self.inference_time = tk.StringVar(value="Inference time: --")
        self.review_text = tk.StringVar(value="Select an image and run the assessment.")
        ttk.Label(result_frame, text="Model classification", style="Subtitle.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.result_class, style="Result.TLabel").pack(anchor="w", pady=(4, 18))
        ttk.Label(result_frame, textvariable=self.fake_score).pack(anchor="w")
        self.fake_bar = ttk.Progressbar(result_frame, maximum=100, mode="determinate")
        self.fake_bar.pack(fill="x", pady=(4, 14))
        ttk.Label(result_frame, textvariable=self.real_score).pack(anchor="w")
        self.real_bar = ttk.Progressbar(result_frame, maximum=100, mode="determinate")
        self.real_bar.pack(fill="x", pady=(4, 18))
        ttk.Separator(result_frame).pack(fill="x", pady=8)
        ttk.Label(result_frame, textvariable=self.inference_time).pack(anchor="w", pady=(4, 10))
        ttk.Label(result_frame, text="Review guidance", style="Section.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.review_text, wraplength=390, justify="left").pack(anchor="w", fill="x", pady=(6, 12))
        self.analyse_button = ttk.Button(result_frame, text="Analyse Image", style="Action.TButton", command=self.analyse_image, state="disabled")
        self.analyse_button.pack(anchor="w", pady=(8, 0))
        ttk.Label(self.assessment_tab, text="WorkShield provides preliminary image classification only. A prediction does not verify a person's identity.", wraplength=900).grid(row=2, column=0, columnspan=2, sticky="w", pady=(14, 0))

    def _build_analytics_tab(self):
        ttk.Label(self.analytics_tab, text="Model Performance", style="Section.TLabel").pack(anchor="w", pady=(0, 12))
        self.analytics_text = tk.Text(self.analytics_tab, height=22, wrap="word", font=("Consolas", 10))
        self.analytics_text.pack(fill="both", expand=True)
        self.analytics_text.insert("1.0", "No evaluation report found yet.\n\nTrain the model and run:\npython -m Python.evaluate_model\n\nThe report will appear here.")
        self.analytics_text.configure(state="disabled")
        ttk.Button(self.analytics_tab, text="Refresh Evaluation Results", command=self.refresh_analytics).pack(anchor="w", pady=(12, 0))

    def _build_about_tab(self):
        content = (
            "WORKSHIELD\nAI-Powered Remote Identity and Workstation Verification\n\n"
            "CURRENT IMPLEMENTED SCOPE\n"
            "The current prototype uses a VGG16-based deep-learning model to classify a submitted facial image as REAL or FAKE. The result is an initial screening indicator intended to support human review.\n\n"
            "LIMITATIONS\n"
            "• A REAL prediction does not prove that an image is authentic.\n"
            "• A FAKE prediction does not prove fraud or wrongdoing.\n"
            "• The model does not match a face to an identity.\n"
            "• The model does not establish that a person is physically present.\n"
            "• The model does not perform dynamic liveness detection.\n"
            "• The application does not monitor or control workstation access.\n\n"
            "FUTURE EXTENSIONS IN THE PROJECT BRIEF\n"
            "Remote identity verification, video-based liveness detection, speech/audio verification, an HR guidance assistant and workstation access integration are future development areas.\n\n"
            "PRIVACY AND RESPONSIBLE USE\n"
            "Facial images may contain personal information. Use only data that the project is authorised to process. Consider POPIA, purpose limitation, lawful processing, consent where required, access controls, data minimisation and retention requirements.\n\n"
            "Do not use this prototype as the sole basis for recruitment, disciplinary, identity or access-control decisions."
        )
        text = tk.Text(self.about_tab, wrap="word", font=("Segoe UI", 10), height=25)
        text.pack(fill="both", expand=True)
        text.insert("1.0", content)
        text.configure(state="disabled")

    def _load_model_background(self):
        try:
            model, metadata = load_workshield_model()
            self.root.after(0, lambda: self._model_loaded(model, metadata, None))
        except Exception as error:
            message = str(error)
            self.root.after(0, lambda: self._model_loaded(None, None, message))

    def _model_loaded(self, model, metadata, error):
        if error:
            self.model_status.set("Model not available. Train WorkShield first.")
            self.analyse_button.configure(state="disabled")
            return
        self.model = model
        self.metadata = metadata
        self.model_status.set("WorkShield model loaded and ready.")
        self.analyse_button.configure(state="normal")

    def select_image(self):
        selected = filedialog.askopenfilename(
            title="Select a facial image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not selected:
            return
        self.selected_image = Path(selected)
        try:
            image = Image.open(self.selected_image).convert("RGB")
            image.thumbnail((440, 440), Image.Resampling.LANCZOS)
            self.preview_reference = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_reference, text="")
            self.result_class.set("Awaiting analysis")
            self.fake_score.set("FAKE model score: --")
            self.real_score.set("REAL model score: --")
            self.fake_bar["value"] = 0
            self.real_bar["value"] = 0
            self.inference_time.set("Inference time: --")
            self.review_text.set("Image selected. Run the assessment when ready.")
        except Exception as error:
            messagebox.showerror("Image Error", f"Could not open this image:\n{error}")

    def clear_image(self):
        self.selected_image = None
        self.preview_reference = None
        self.preview_label.configure(image="", text="No image selected\n\nChoose a facial image to begin.")
        self.result_class.set("Awaiting image analysis")
        self.fake_score.set("FAKE model score: --")
        self.real_score.set("REAL model score: --")
        self.fake_bar["value"] = 0
        self.real_bar["value"] = 0
        self.inference_time.set("Inference time: --")
        self.review_text.set("Select an image and run the assessment.")

    def analyse_image(self):
        if self.selected_image is None:
            messagebox.showwarning("No Image", "Select an image before running an assessment.")
            return
        if self.model is None:
            messagebox.showerror("Model Not Ready", "The trained WorkShield model is not available.")
            return
        self.analyse_button.configure(state="disabled")
        self.result_class.set("Analysing...")
        self.root.update_idletasks()
        try:
            result = predict_loaded_model(self.model, self.selected_image, self.metadata)
            self.result_class.set(result["predicted_class"])
            self.fake_score.set(f"FAKE model score: {result['fake_score'] * 100:.2f}%")
            self.real_score.set(f"REAL model score: {result['real_score'] * 100:.2f}%")
            self.fake_bar["value"] = result["fake_score"] * 100
            self.real_bar["value"] = result["real_score"] * 100
            self.inference_time.set(f"Inference time: {result['inference_seconds']:.4f} seconds")
            self.review_text.set(result["review_note"])
        except Exception as error:
            messagebox.showerror("Assessment Error", str(error))
            self.result_class.set("Assessment failed")
        finally:
            self.analyse_button.configure(state="normal")

    def refresh_analytics(self):
        self.analytics_text.configure(state="normal")
        self.analytics_text.delete("1.0", "end")
        if not EVALUATION_JSON_PATH.exists():
            self.analytics_text.insert("1.0", "No evaluation report found.\n\nRun:\npython -m Python.evaluate_model")
            self.analytics_text.configure(state="disabled")
            return
        try:
            with open(EVALUATION_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            lines = [
                "WORKSHIELD EVALUATION RESULTS", "=" * 42,
                f"Evaluation split: {data['evaluation_split']}",
                f"Images evaluated: {data['number_of_images']}",
                f"Accuracy: {data['accuracy'] * 100:.2f}%",
                f"Threshold: {data['threshold']:.2f}", "",
                "CLASSIFICATION REPORT"
            ]
            for name, metrics in data["classification_report"].items():
                if isinstance(metrics, dict):
                    lines.append(
                        f"{name}: precision={metrics.get('precision', 0):.3f}, "
                        f"recall={metrics.get('recall', 0):.3f}, "
                        f"F1={metrics.get('f1-score', 0):.3f}"
                    )
            lines.extend([
                "", "CONFUSION MATRIX", str(data["confusion_matrix"]), "",
                f"FAKE images predicted REAL: {data['fake_images_predicted_as_real']}",
                f"REAL images predicted FAKE: {data['real_images_predicted_as_fake']}",
                f"Average prediction time: {data['average_seconds_per_image']:.6f} seconds/image"
            ])
            if data.get("roc_auc_real_as_positive") is not None:
                lines.append(f"ROC-AUC: {data['roc_auc_real_as_positive']:.4f}")
            if "NOT AN INDEPENDENT TEST" in data["evaluation_split"]:
                lines.extend(["", "WARNING: These are validation results, not independent test results."])
            self.analytics_text.insert("1.0", "\n".join(lines))
        except Exception as error:
            self.analytics_text.insert("1.0", f"Could not read evaluation report:\n{error}")
        self.analytics_text.configure(state="disabled")

def launch_app():
    root = tk.Tk()
    WorkShieldApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_app()
