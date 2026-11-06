# Computer-Vision-Project
In the following 3 weeks, you will be building on your previous work with image datasets, and focus on applying the YOLO (You Only Look Once)framework to your custom dataset for your chosen task(classfication, detection, segmentation). In this tutorial we will focus on object detection only, for other tasks you will have to make few changes in notebook as well as labelled files, because each task requires specific lableling format and pre-trained models. You will fine-tune the YOLO model using the dataset you've created or annotated.Then focus on evaluating the YOLO model. You will assess its performance, discuss a potential use-case, and critically analyze its shortcomings and ethical considerations. 

 

 

**Tasks:** 
- Clone this repository
- Integrate your custom dataset into the YOLO framework (transfer learning).
- Fine-tune the YOLO model using your dataset. This involves adjusting the model's parameters to better suit your specific data.
- Conduct evaluation of your fine-tuned YOLO model. Use appropriate metrics to assess its performance (e.g., precision, recall, accuracy, F1 score, etc.).

---

# Group 6 setup: gesture-controlled photo booth

YOLOv8s fine-tuned to detect four hand gestures (palm, fist, like, dislike). The images and labels live in the shared Google Drive folder **ComputerVisionProject-Group6-Dataset**; the code reads them from there through the Google Drive desktop app, so nothing has to be downloaded by hand.

## 1. Get the code
```
git clone https://github.com/Amynguyen2424/Computer-Vision-Project-Group6.git
cd Computer-Vision-Project-Group6
git checkout Ben
```

## 2. Install Google Drive for desktop
1. Download and install it from https://www.google.com/drive/download/ and sign in with the Google account that has access to the group folder.
2. In Google Drive in your browser, open **Shared with me**, right-click **ComputerVisionProject-Group6-Dataset** and choose **Organize → Add shortcut → My Drive**. The desktop app only sees shared folders that have a shortcut in your Drive.
3. Check that the folder shows up: on Windows under the Google Drive drive (usually `G:\My Drive`), on Mac under **Google Drive** in Finder.
4. Optional, makes the first build faster: right-click the folder there and choose **Offline access → Available offline** (about 1.4 GB).

## 3. Set up Python
You need Python 3.10 or newer. In the project folder:

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac
```

Install PyTorch first:
- **NVIDIA GPU (Windows):** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- **No NVIDIA GPU, or Mac:** `pip install torch torchvision`

Then the rest, and register the notebook kernel:
```
pip install -r requirements.txt
python -m ipykernel install --user --name cv-group6 --display-name "Python (CV Group 6)"
```

## 4. Build the dataset from Drive
```
python prepare_dataset.py
```
It finds the Drive folder, checks every image/label pair (and fixes Joseph's class order), and copies everything into `dataset/` (ignored by git). It should end with **train 1280 / val 160 / test 160** and **128** team photos. The first run takes about 10-15 minutes because Drive fetches each image; later runs are much faster. Run it again whenever someone changes photos or labels on Drive.

## 5. Run the notebooks
```
jupyter lab
```
Open a notebook, check that the kernel (top right) is **Python (CV Group 6)**, then **Run → Run All Cells**.

| File | What it does |
|---|---|
| `Gesture_YOLO_Baseline.ipynb` | Trains and evaluates the baseline (about 8.5 min on an RTX 5060 Ti, much longer on a CPU) |
| `Gesture_YOLO_Tuning_imgsz.ipynb` | Ben's hyperparameter, `imgsz` 480 / 640 / 800 / 960. Uses the saved models in `models/` and only re-scores them, unless you delete a model to retrain it |
| `gesture_eval.py` | Baseline hyperparameters (`BASELINE`) and the evaluation every run uses. For your own hyperparameter, train with `{**BASELINE, "<name>": value}` and score with `evaluate()` and `error_analysis()` |
| `prepare_dataset.py` | Builds `dataset/` from the Drive folder |
| `report/Baseline_Evaluation_Report.pdf` | Detailed baseline evaluation. Rebuild it with `python report/build_baseline_report.py` (Windows, needs Microsoft Edge) |
| `models/`, `results/` | Saved weights and the evaluation output of every run |

Our own photos (numbers 369-400 of each class) are the test set, so always compare models on the **team-only test** numbers, not on validation. 
