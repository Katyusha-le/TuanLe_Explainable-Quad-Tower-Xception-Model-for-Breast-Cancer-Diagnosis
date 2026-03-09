# Explainable-Quad-Tower-Xception-Model-for-Breast-Cancer-Diagnosis
This project is an Explainable Multimodal Diagnostic AI that classifies breast cancer into 5 subtypes using 2D mammograms and clinical metadata . Based on a Quad-Tower Xception architecture, it integrates XAI tools (GRAD-CAM++, SHAP, TCAV) to eliminate the "black box" effect, providing transparent clinical decision support.

## I. Introduction
Breast cancer diagnosis and treatment planning require analyzing highly complex and heterogeneous data. A critical task is identifying the molecular subtype of the tumor such as Benign, Luminal A, Luminal B, HER2-enriched, and Triple-Negative as this heavily dictates the treatment strategy. While Deep Convolutional Neural Networks (DCNNs) are promising for medical imagery, they are often seen as "black boxes" lacking clinical transparency. 

This project introduces an **Explainable Multimodal Diagnostic AI Framework**. By fusing 2D mammography images with patient clinical metadata (age and lesion abnormality), the framework predicts the cancer subtype while heavily integrating state-of-the-art Explainable AI techniques. This ensures that medical professionals can interpret *why* the model makes specific predictions, bridging the gap between deep learning and clinical decision support.

## Data Source (CMMD Dataset)

The dataset used for this project is the **CMMD (The Chinese Mammography Database)**. It is a publicly available secondary dataset hosted by **TCIA (The Cancer Imaging Archive)** and provided by Cui et al., (2021). 

Here are the key details regarding the data origin and its composition:

*   **Origin & Collection:** The data was collected from breast cancer patients in China between July 2012 and January 2016.
*   **Raw Data:** The original dataset contains 5,202 high-resolution 2D mammography images (2,294 × 1,914 pixels) and clinical records from 1,775 patients.
*   **Inclusion Criteria:** Because this Quad-Tower architecture strictly requires multimodal inputs, any image lacking complete accompanying clinical metadata (Age, Abnormality type, and Subtype label) was filtered out. 

### Final Dataset Breakdown
After cleaning and filtering, a total of **4,056 images** were retained for the training, validation, and testing splits. 

**Clinical Metadata Features Used:**
*   **Age:** Patient ages ranging from 17 to 87 years old (Mean age: 46.95).
*   **Abnormality Type:** Classified into three categories: Mass, Calcification, or Both.

**Class Distribution (4,056 total images):**
*   **Benign:** 1,108 images 
*   **Luminal A:** 598 images 
*   **Luminal B:** 1,476 images 
*   **HER2-enriched (HER2+):** 530 images 
*   **Triple-Negative:** 344 images

*(Note 1: Please go to https://www.cancerimagingarchive.net/ for data usage policy and guidance)*

*(Note 2: Due to the high class imbalance such as Luminal B having 1,476 images while Triple-Negative has only 344, the training strategy implements weighted random sampling and a Class-Weighted Focal Loss to ensure the minority classes are learned effectively)*.

---

## II. Model Architecture
The core of this system is a **Quad-Tower architecture** built on the Xception backbone, utilizing early data enhancement, deep supervision, and late multimodal fusion.

<img width="3187" height="595" alt="Model diagram" src="https://github.com/user-attachments/assets/42938fdb-1917-429b-937b-e9eb5918beef" />

### 1. Data Preprocessing & Channel Stacking
The data preprocessing pipeline, and particularly the **Ensemble Cropping mechanism**, is a sophisticated, annotation-free system designed to automatically locate and extract the most suspicious Region of Interest (ROI) from a mammogram.

#### 1.1. Robust Scaling & Artifact Removal
Before any cropping occurs, the raw DICOM images undergo **percentile-based clipping**. Mammograms often contain extremely bright metal artifacts (like patient tags or clips) that would skew the image contrast if standard Min-Max scaling were used. The script calculates the **99th percentile (P99)** of the pixel intensities and clips any outlier pixels above this value. The image is then safely normalized to a range and converted to an 8-bit `uint8` format to keep storage efficient and contrast high.

#### 1.2. The Ensemble Cropping Architecture
The `crop_tumor_ensemble` function acts as a multi-stage search engine to locate the tumor core without human intervention. 

<img width="1145" height="1189" alt="Ensemble cropping" src="https://github.com/user-attachments/assets/a33fe155-ac70-44b5-9cd8-448c440488b3" />

***Phase 1: Defining the "Safe Zone" (Erosion):***
Mammograms often have a bright, thick skin line that algorithms mistake for tumors. The script applies an Otsu threshold to create a binary mask of the entire breast, and then uses an erosion operation (`cv2.erode`). The erosion kernel size is dynamically calculated as **1.5% of the image's shortest edge**. This dynamically "shaves off" the skin and border artifacts regardless of the image resolution, leaving a clean "safe zone" consisting only of inner breast tissue.

***Phase 2: Gravity Pull & Focus Zone:***
To prevent the algorithms from searching irrelevant areas, the script calculates the image's "center of gravity" using spatial moments (`cv2.moments`) based on pixel intensity. Because tumors and dense tissues are brighter (heavier) and fat is darker (lighter), the calculated center naturally pulls away from fatty areas and anchors toward the densest part of the breast. A circular "Focus Zone" with a radius equal to 1/3 of the image dimensions is then drawn around this anchor.

***Phase 3: The "Smart Score" Evaluator:***
When algorithms detect potential abnormalities, they need a way to rank them. Instead of just picking the largest or brightest blob, the system uses a hybrid logarithmic scoring function:
**`Score = Brightness × ln(Area + 1)`**

Using the natural logarithm (`np.log1p`) is a crucial design choice: it prevents massive but dim areas (like the pectoral muscle) from overwhelming the score, while heavily favoring small, extremely bright spots (which are classic signs of malignancies like microcalcifications or tumor cores).

***Phase 4: The 3 Expert Voters:***
Inside the Focus Zone, three independent computer vision algorithms act as a "committee" to propose the exact center of the tumor:
1.  **Voter 1 - Otsu Thresholding:** This voter searches for the brightest pixel clusters. It is highly effective at identifying large, solid tumor masses with clear boundaries.
2.  **Voter 2 - Spectral Residual Saliency:** Using Fast Fourier Transform (FFT), this algorithm analyzes the image in the frequency domain. It filters out repeating "background" frequencies (like normal tissue) to isolate structural anomalies and sudden spikes. It is an expert at finding hidden, low-contrast lesions or structural distortions that Otsu might miss.
3.  **Voter 3 - Adaptive Difference of Gaussians (DoG):** This acts as a band-pass filter by subtracting a heavily blurred image (9x9 Gaussian) from a lightly blurred one (5x5 Gaussian). This perfectly isolates high-frequency details, making it the primary expert for detecting tiny microcalcifications and spiculated (star-like) tumor margins.

***Phase 5: Consensus & Final Cropping:***
Once the three voters propose their coordinates, the system calculates the Euclidean distance between them to find a consensus. 
*   **Agreement:** If any two voters select points within a dynamic tolerance distance (at least 75 pixels or 10% of the image size), the algorithm averages their coordinates to find the highly confident final center.
*   **Fallback Hierarchy:** If the voters completely disagree (a highly complex or vague image), the system does not guess randomly It falls back on a strict hierarchy based on clinical reliability: **Saliency > Otsu > DoG > Gravity Anchor**.

Finally, the script crops a 299x299 Region of Interest (ROI) around this consensus center, generating the "Local" view. This meticulously extracted crop is then fed into the local processing towers (Sigmoid and CLAHE) of the Quad-Tower model.

To maximize feature extraction from mammograms, the model processes images into four parallel streams (towers), acting as four distinct viewpoints:
*   **Tower 1 (Global Gamma):** Whole breast image applying Gamma corrections (0.6 and 1.5) to stretch dark regions and compress bright pixels.

<img width="1154" height="429" alt="Channel 1" src="https://github.com/user-attachments/assets/2cb39932-8b38-4a1a-85df-ae5d8e8e7344" />

*   **Tower 2 (Global CLAHE):** Whole breast image applying CLAHE (Clip Limit 2.0 & 4.0) to highlight micro-tentacles and spiculated margins.

<img width="1154" height="396" alt="Channel 2" src="https://github.com/user-attachments/assets/859a02e8-934c-4e90-8e39-f8d19854874f" />

*   **Tower 3 (Local Sigmoid):** Cropped Region of Interest (ROI) applying heavy Sigmoid curves to isolate high-density tumor cores.

<img width="1154" height="400" alt="Channel 3" src="https://github.com/user-attachments/assets/fbe0b416-2228-46a2-88d0-3c50529201d2" />

*   **Tower 4 (Local CLAHE):** Cropped ROI applying CLAHE for local texture enhancement.

<img width="1154" height="398" alt="Channel 4" src="https://github.com/user-attachments/assets/0d352a36-8396-40fc-afb3-039f726d3442" />

### 2. Feature Extraction & Expert Processing Blocks
Each stream is fed into an independent **Xception backbone** (pre-trained on ImageNet). The resulting 2048-dimensional spatial tensors pass through a funneling **Expert Processing Block** (2048 -> 1024 -> 512 dimensions). This block employs a self-attention mechanism (Sigmoid gating) to filter out noise and amplify critical morphological signals before routing them to the next stage. An auxiliary classifier (`aux_out`) is attached to each block to enforce deep supervision.

### 3. Cross-Attention Mechanism
Instead of isolating the four streams until the end, a **Cross-Attention Mechanism** allows each tower to query and borrow highly relevant morphological features from the other three towers (acting as a "medical consultation board"). 

### 4. Multimodal Fusion & Classification Head
The four enriched 512D image vectors are concatenated with a 64D clinical metadata vector (Age and Abnormality) to form a unified 2112D vector. A **Squeeze-and-Excitation (SE)** mechanism recalibrates this vector globally, turning up the "volume" of highly correlated multimodal signals while suppressing noise. Finally, a deep Classification Head outputs the Softmax probabilities for the 5 breast cancer subtypes.

### 5. Training Strategy
*   **Class-Weighted Focal Loss (γ=2.5):** Mitigates heavy class imbalance and forces the network to focus on hard-to-classify samples without destroying gradients.
*   **Weighted Random Sampling:** Ensures every training batch dynamically maintains a balanced class ratio by sampling using inverse frequencies.
*   **Fine-Tuning Strategy:** Freezes the base Xception model and only unfreezes **Block 14** (the deepest semantic layer) with a micro learning rate (1e-6) while keeping Batch Normalization frozen to prevent weight destruction.

---

## III. Model Explainability
To eliminate the "black box" nature of DCNNs, the pipeline incorporates several interpretability tools:

*   **Ensemble Cropping (Preprocessing):** Before training, local ROIs are cropped using an ensemble of 3 algorithms: Otsu Thresholding, Spectral Residual Saliency, and Adaptive Difference of Gaussians (DoG). A consensus mechanism uses an intensity-area logarithmic scoring function (`Score = I × ln(Area + 1)`) to isolate the tumor from background noise without human annotation.

<img width="1145" height="1189" alt="Ensemble cropping" src="https://github.com/user-attachments/assets/2602ee33-6828-43e7-8ec9-c6e161e25dcf" />

*   **GRAD-CAM++:** Applied natively to the four spatial image streams at the Global Average Pooling bottleneck. It tracks higher-order spatial derivatives to generate highly accurate heatmaps, showing exactly which regions the network focused on for its prediction.

<img width="2203" height="985" alt="Implementing GRAD-CAM++" src="https://github.com/user-attachments/assets/f24a9b45-3507-4c17-88cd-0c4ad88bcca3" />

In a case by case analysis, the result will be presented like this.

<img width="1989" height="1180" alt="GRAD-CAM++ case by case" src="https://github.com/user-attachments/assets/74102c44-12e7-4e58-afba-13cf38898e70" />

*   **SHAP (SHapley Additive exPlanations):** Used heavily to demystify the clinical data. It provides waterfall plots showing exactly how a patient's age or specific lesion type shifted the prediction percentage toward or away from a specific malignancy.

<img width="2011" height="1011" alt="SHAP text 1" src="https://github.com/user-attachments/assets/9085c816-7e61-405f-82be-e896ab086264" />
<img width="2011" height="1018" alt="SHAP text 2" src="https://github.com/user-attachments/assets/7bd98442-9d4e-42f8-83c7-6e23825eba8b" />

This SHAP analysis can be applied to image data too.

<img width="2037" height="742" alt="SHAP image" src="https://github.com/user-attachments/assets/cbd068c9-72f7-4f3e-b7ab-4a64614833e2" />

*   **TCAV (Testing with Concept Activation Vectors):** Translates neural network logic into high-level human concepts. It performs directional derivative calculations to statistically prove whether the model learned clinical hypotheses—for example, confirming if the model associates "Age > 60" with a higher probability of "Triple Negative" breast cancer.

<img width="1279" height="638" alt="TCAV hypothesis" src="https://github.com/user-attachments/assets/7009a85a-d65b-4349-bf1e-c25bc6e8197d" />

---

## IV. Result & Comparison

Classifying breast cancer into 5 subtypes using 2D mammograms is an extremely challenging task due to visually identical traits among HR+ subtypes (e.g., Luminal A vs. Luminal B). The fine-tuned Quad-Tower Xception model achieved the following global metrics on the test set:

*   **Accuracy (ACC):** 32.03%
*   **Macro AUC:** 0.6468
*   **Macro F1:** 0.3093
*   **Sensitivity (SENS):** 32.79%
*   **Specificity (SPEC):** 82.36%

<img width="1489" height="1324" alt="Confusion matrix" src="https://github.com/user-attachments/assets/d5152262-3408-4def-a252-5664ac97a5df" />

While the overall accuracy appears modest, evaluating the model's behavior per class reveals its clinical reliability:

### Comparison with Alternative Backbones (DenseNet121 & ResNet50V2)
When tested with DenseNet121 and ResNet50V2 backbones, the competitors achieved slightly higher overall ACC (40.10% and 37.41% respectively). However, they fell into the trap of over-optimizing for majority classes:
*   **The Triple-Negative Failure:** Both DenseNet121 and ResNet50V2 nearly completely failed to identify the highly malignant Triple-Negative class (DenseNet F1: 0.000, ResNet F1: 0.049). 
*   **The Xception Advantage:** The Xception backbone maintained strong feature extraction across the entire spectrum. It actively preserved diagnostic capability for the minority Triple-Negative class (F1: 0.242 before fine-tuning, 0.218 after fine-tuning) while balancing the remaining categories. 

### Comparison with Previous Research
When compared to a recent 5-class mammography prediction model, **DenseNet121-CBAM** (Luo et al., 2025), the Quad-Tower Xception model shows a distinct advantage in clinical safety. 
*   While Luo's model achieved a slightly higher AUC (0.6494 vs 0.6414) and SENS (45.5% vs 31.3%), it suffered from a massive drop in Specificity (58.4%). 
*   Our fine-tuned Xception model maintains a **Specificity of >82.3%**, drastically reducing the rate of false-positive predictions across multiple classes, establishing it as a much safer tool for clinical exclusion.

| Stats | Quad-Stream Xception | DenseNet121-CBAM (Luo et al., 2025) |
| --- | --- | --- |
| AUC | 0.6468 | 0.6494 |
| ACC (%) | 0.3203 | 0.325 |
| F1 | 0.3093 | - |
| SENS (%) | 0.3279 | 0.455 |
| SPEC (%) | 0.8236 | 0.584 |

---

## V. Installation and Usage Guide
This project is optimized to run on **Google Colab** to leverage its free GPU resources and seamless integration with Google Drive for storage.

### 1. Prerequisites & Environment Setup
- Upload and open the provided Jupyter Notebook (`.ipynb` file) in Google Colab.
- Mount your Google Drive when prompted, as the processed data and model checkpoints will be saved there to prevent data loss when the runtime disconnects [1, 2].
- Install the required dependencies. The notebook includes an execution cell to automatically install these packages:
  ```python
  !pip install pydicom opencv-python tqdm scikit-learn tensorflow openpyxl
  ``` [3]

### 2. Data Acquisition (TCIA Database)
The project utilizes the CMMD dataset from **The Cancer Imaging Archive (TCIA)**. Due to TCIA's data-sharing policies, you cannot host the raw data on GitHub. The acquisition process is split into an automated image download and a manual metadata upload.

*   **Image Data (Automated):** You do not need to download the gigabytes of image data manually. The notebook contains a script that calls the TCIA API to fetch and extract the DICOM mammography images directly into the Colab environment (`/content/cmmd_data`).
*   **Clinical Metadata (Manual Action Required):** 
    1. Go to the [TCIA CMMD page](https://www.cancerimagingarchive.net/).
    2. Download the clinical metadata file (Excel/CSV format) which contains patient IDs, ages, abnormality types, and subtypes.
    3. In the Colab notebook, run the metadata upload cell (`files.upload()`) and select the downloaded file from your local machine.

### 3. Data Preprocessing
Once the data is in the Colab environment, run the preprocessing pipeline cells. This step will:
- Execute the **Ensemble Cropping** mechanism to automatically locate and extract the Region of Interest (ROI) from the full mammograms.
- Perform robust percentile-based scaling to remove metal artifacts.
- Normalize patient ages and one-hot encode the abnormality types.
- Split the dataset into Training, Validation, and Testing sets.
- **Save to Drive:** The script will convert the arrays into `.npy` files and copy them to your mounted Google Drive (default path: `/content/drive/MyDrive/CMMD_preprocessed/v1.0`).

### 4. Model Training
To safely train the Quad-Tower architecture and prevent catastrophic forgetting of the pre-trained weights, the training is executed in two phases:

*   **Phase 1 (Feature Extraction):** The Xception backbones are completely frozen. The model trains the dense layers, cross-attention mechanisms, and fusion blocks using a Class-Weighted Focal Loss (γ=2.5) to handle class imbalance. Checkpoints are saved continuously to your Drive.
*   **Phase 2 (Fine-Tuning):** The script unfreezes **Block 14** (the top-most semantic layer of the Xception towers) while keeping Batch Normalization layers frozen. It recompiles with a micro learning rate (`1e-6`) to carefully refine the spatial feature extraction specifically for breast cancer diagnosis.

### 5. Evaluation & Explainable AI (XAI)
After the training phases are complete, run the evaluation cells to analyze the model's performance and interpret its decisions:
- **Global Metrics:** Generate the Confusion Matrix, Macro AUC, F1-scores, Sensitivity, and Specificity reports.
- **GRAD-CAM++:** Run the interactive widget to visualize the spatial heatmaps. It intercepts the `GlobalAveragePooling2D` layer to show exactly which parts of the mammogram the 4 image towers focused on.
- **SHAP (Tabular/Metadata):** Generates waterfall and summary plots to quantify how the clinical metadata shifted the model's final prediction.
- **TCAV (Concept Testing):** Utilizes a custom `Fast_TCAV` engine to statistically test human-understandable concepts against the model's latent fusion space.

---

## VI. Reference
Cui, Chunyan; Li Li; Cai, Hongmin; Fan, Zhihao; Zhang, Ling; Dan, Tingting; Li, Jiao; Wang, Jinghua. (2021) The Chinese Mammography Database (CMMD): An online mammography database with biopsy confirmed types for machine diagnosis of breast. The Cancer Imaging Archive. DOI: https://doi.org/10.7937/tcia.eqde-4b16
Luo, Y., Wei, J., Gu, Y., Zhu, C., & Xu, F. (2025). Predicting molecular subtype in breast cancer using deep learning on mammography images. Frontiers in Oncology, 15, 1638212. https://doi.org/10.3389/fonc.2025.1638212
