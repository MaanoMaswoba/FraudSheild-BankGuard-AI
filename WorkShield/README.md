# WorkShield
## AI-Powered Remote Identity and Workstation Verification

WorkShield is a Python desktop prototype for preliminary REAL/FAKE classification of facial images using transfer learning with VGG16. This implementation follows the current scope in the WorkShield Business Analysis project brief. It does not claim to verify a person's identity, detect a live person, perform liveness detection, monitor a workstation or control access.

---

## 1. Current implemented scope

The prototype provides:
1. A labelled facial-image dataset structure.
2. Image loading and resizing to 224 × 224 RGB.
3. Training-time image augmentation.
4. VGG16 ImageNet feature extraction.
5. A binary REAL/FAKE classification head.
6. Model training with validation monitoring.
7. Early stopping and learning-rate reduction.
8. Saved model and class metadata.
9. Individual-image prediction through a command-line script.
10. A Tkinter desktop application for image assessment.
11. Evaluation metrics, confusion matrix and error counts.
12. Training history and evaluation reports.

### Future extensions, not currently implemented

The brief identifies remote identity verification, dynamic liveness detection using video, speech/audio verification, an HR guidance assistant and workstation access verification as possible future development. These must not be presented as completed functionality.

---

## 2. Project structure

```text
WorkShield/
├── Data/
│   ├── raw/
│   │   ├── fake/
│   │   └── real/
│   └── rvf10k/
│       ├── train/fake/
│       ├── train/real/
│       ├── valid/fake/
│       ├── valid/real/
│       ├── test/fake/
│       └── test/real/
├── Models/
├── Results/
├── Python/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── model_components.py
│   ├── prepare_dataset.py
│   ├── check_dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── test.py
│   └── evaluate_model.py
├── app.py
├── main.py
├── requirements.txt
├── setup_windows.ps1
├── .gitignore
└── README.md
```

The `.venv` folder is created locally during setup. It is not packaged because virtual environments are machine-specific. Dataset images and trained model weights are also not included in the archive; add the authorised dataset before training.

---

## 3. System requirements

Recommended: Windows 10/11, Python 3.11 (64-bit), 16 GB RAM, sufficient disk space and an internet connection for package installation and first-time VGG16 ImageNet weight download. The project runs on CPU. Native Windows TensorFlow may not use an NVIDIA GPU with current TensorFlow releases; GPU acceleration requires a separately configured compatible WSL2/Linux CUDA environment.

---

## 4. Create the virtual environment

Open the WorkShield folder in VS Code, then open Terminal → New Terminal. Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

The script creates `.venv`, upgrades pip and installs `requirements.txt`. Alternatively:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

---

## 5. Add the dataset

The brief describes 10,000 images: 7,000 training and 3,000 validation. If the dataset is already split, place it in:

```text
Data/rvf10k/train/fake/
Data/rvf10k/train/real/
Data/rvf10k/valid/fake/
Data/rvf10k/valid/real/
```

An independent test split is recommended at `Data/rvf10k/test/fake/` and `Data/rvf10k/test/real/`. Do not treat validation data as an independent test set.

If all images are in two unsplit folders, place them in `Data/raw/fake/` and `Data/raw/real/`, then run:

```powershell
.\.venv\Scripts\python.exe -m Python.prepare_dataset
```

This creates a stratified 70/15/15 training/validation/test split. This is an alternative to the pre-existing 7,000/3,000 split. Check the folders with:

```powershell
.\.venv\Scripts\python.exe -m Python.check_dataset
```

Both classes must contain images in training and validation.

---

## 6. Stage 1 — Train the model

```powershell
.\.venv\Scripts\python.exe -m Python.train
```

The training pipeline checks the folders, loads labelled images, resizes them to 224 × 224 RGB, applies training-only augmentation, performs VGG16 preprocessing inside the model, loads pretrained VGG16, freezes its convolutional base and trains a custom classification head. It uses binary cross-entropy, Adam, class weights, early stopping, learning-rate reduction and best-validation-loss checkpointing.

No manual `steps_per_epoch` or `validation_steps` are supplied; TensorFlow derives batch counts from the datasets. This avoids the input-exhaustion problem identified in the brief.

Outputs:
- `Models/workshield_vgg16.keras`
- `Models/class_mapping.json`
- `Results/training_history.json`
- `Results/training_history.png`

Class mapping is fixed: FAKE = 0, REAL = 1. The sigmoid output represents the REAL class score. Default batch size is 16 for a memory-conscious start; set `WORKSHIELD_BATCH_SIZE=32` if the computer can handle it.

---

## 7. Stage 2 — Evaluate the model

```powershell
.\.venv\Scripts\python.exe -m Python.evaluate_model
```

The evaluator uses `test` when available. Otherwise it uses validation data and labels it clearly. Explicit split options:

```powershell
.\.venv\Scripts\python.exe -m Python.evaluate_model --split test
.\.venv\Scripts\python.exe -m Python.evaluate_model --split valid
```

Metrics include accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC where both classes exist, class-specific misclassification counts, average prediction time, model size and process-memory readings. Reports are saved in `Results/`. Validation metrics are not independent test results.

---

## 8. Stage 3 — Test one image

```powershell
.\.venv\Scripts\python.exe -m Python.test "Data/sample.jpg"
```

Replace the sample path with the path to an image. The script reports the predicted class, FAKE and REAL model scores, selected class score, decision threshold, inference time and review guidance. Model scores are not calibrated proof that an image is genuine or manipulated.

---

## 9. Stage 4 — Launch the desktop application

```powershell
.\.venv\Scripts\python.exe main.py
```

The Tkinter application has three tabs:
- **Image Assessment:** select and preview an image, run the classifier and view class scores and review guidance.
- **Model Analytics:** load evaluation results and inspect class-level metrics and confusion-matrix counts.
- **About & Limitations:** explain current scope, future extensions, privacy and responsible-use limits.

The model must be trained before image analysis is available.

---

## 10. Model architecture

```text
Input facial image (224 × 224 × 3)
        ↓
Training augmentation (training only)
        ↓
VGG16 preprocessing
        ↓
Frozen VGG16 ImageNet base
        ↓
Global Average Pooling
        ↓
Dense (256, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (1, Sigmoid)
        ↓
REAL / FAKE assessment
        ↓
Human review guidance
```

Global Average Pooling replaces Flatten to reduce the custom classification head's parameter count. This is an architectural change to evaluate, not a guarantee of higher accuracy. The VGG16 base remains frozen for the baseline; fine-tuning can be tested later as a separate controlled experiment.

---

## 11. Responsible use, privacy and limitations

A REAL result does not prove that an image is authentic. A FAKE result does not prove fraud or wrongdoing. The current model does not perform face-to-person identity matching, identity-document verification, live-person detection, dynamic liveness detection, speech/audio analysis, background workstation monitoring or access control.

Facial images may contain personal information. Assess applicable South African requirements, including POPIA, and document the processing purpose, lawful basis, consent where required, access controls, data minimisation, retention limits and data-subject procedures. Do not commit facial images or sensitive data to GitHub. Do not use this prototype as the sole basis for recruitment, disciplinary, identity or access-control decisions.

---

## 12. Final demonstration checklist

1. Confirm dataset source and labels.
2. Check for duplicate or corrupt images.
3. Record training configuration and model version.
4. Keep the independent test split separate from training and model selection.
5. Examine both types of misclassification.
6. Test images from different quality levels and sources.
7. Assess relevant demographic groups only where lawful, appropriate and supported by reliable labels.
8. Record inference time and hardware details.
9. Keep model, metadata and preprocessing versions together.
10. Document limitations and human-review procedures.

Future model updates should be controlled and re-evaluated rather than replacing the model without comparison.

---

## 13. Useful commands

| Task | Command |
|---|---|
| Check dataset | `python -m Python.check_dataset` |
| Prepare raw dataset | `python -m Python.prepare_dataset` |
| Train model | `python -m Python.train` |
| Evaluate model | `python -m Python.evaluate_model` |
| Test one image | `python -m Python.test "Data/sample.jpg"` |
| Launch application | `python main.py` |

Run commands from the WorkShield project root with the virtual environment active, or use the explicit `.venv\Scripts\python.exe` path.
