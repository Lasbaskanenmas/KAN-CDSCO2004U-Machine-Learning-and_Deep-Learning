# KAN-CDSCO2004U - Machine Learning and Deep Learning

## Skin Cancer Detection using Machine Learning

### Abstract

This project investigates the use of machine learning and deep learning for the classification of malignant skin cancer using dermoscopic images. We implemented four models: a custom Convolutional Neural Network (CNN), a pre-trained EfficientNet with transfer learning, a Vision Transformer (ViT), and an ensemble of Gradient Boosted Trees (GBT) trained on structured metadata. All models were trained on the same dataset of 401,059 images from the ISIC archive, alongside metadata with 55 features. To address the 0.1% class imbalance, data augmentation and class weighting were employed. The ensemble model achieved the highest F1-score of 0.59, while ViT correctly identified 64% of malignant cases, making it the most clinically promising model.

### Project Purpose

The objective is to support dermatologists in early identification of malignant skin lesions to improve treatment outcomes and enable scalable triage in regions with limited access to specialists.

### Environment Setup

Python Version: **3.11.12**
Numpy Version: **1.24.4** (crucial for compatibility)

We recommend creating a Miniforge environment:

```bash
conda create -n skin_cancer_env python=3.11.12 numpy=1.24.4 -c conda-forge
conda activate skin_cancer_env
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install absl-py==2.2.2 asttokens==3.0.0 astunparse==1.6.3 catboost==1.2.8 \
certifi==2025.4.26 charset-normalizer==3.4.2 colorama==0.4.6 comm==0.2.2 \
contourpy==1.3.2 cycler==0.12.1 debugpy==1.8.14 decorator==5.2.1 \
executing==2.2.0 flatbuffers==25.2.10 fonttools==4.58.0 gast==0.6.0 \
google-pasta==0.2.0 graphviz==0.20.3 grpcio==1.71.0 h5py==3.13.0 idna==3.10 \
imbalanced-learn==0.13.0 imblearn==0.0 ipykernel==6.29.5 ipython==9.2.0 \
ipython_pygments_lexers==1.1.1 jedi==0.19.2 joblib==1.5.0 jupyter_client==8.6.3 \
jupyter_core==5.7.2 keras==3.9.2 kiwisolver==1.4.8 libclang==18.1.1 \
lightgbm==4.6.0 Markdown==3.8 markdown-it-py==3.0.0 MarkupSafe==3.0.2 \
matplotlib==3.10.3 matplotlib-inline==0.1.7 mdurl==0.1.2 ml_dtypes==0.5.1 \
namex==0.0.9 narwhals==1.39.1 nest-asyncio==1.6.0 numpy==1.24.4 \
opt_einsum==3.4.0 optree==0.15.0 packaging==25.0 pandas==2.2.3 \
parso==0.8.4 pillow==11.2.1 platformdirs==4.3.8 plotly==6.1.0 \
polars==1.29.0 prompt_toolkit==3.0.51 protobuf==5.29.4 psutil==7.0.0 \
pure_eval==0.2.3 pyarrow==20.0.0 Pygments==2.19.1 pyparsing==3.2.3 \
python-dateutil==2.9.0.post0 pytz==2025.2 pywin32==310 pyzmq==26.4.0 \
requests==2.32.3 rich==14.0.0 scikit-learn==1.6.1 scipy==1.15.3 six==1.17.0 \
sklearn-compat==0.1.3 stack-data==0.6.3 tensorboard==2.19.0 \
tensorboard-data-server==0.7.2 tensorflow==2.19.0 \
tensorflow-io-gcs-filesystem==0.31.0 termcolor==3.1.0 \
threadpoolctl==3.6.0 tornado==6.5 traitlets==5.14.3 \
typing_extensions==4.13.2 tzdata==2025.2 urllib3==2.4.0 \
wcwidth==0.2.13 Werkzeug==3.1.3 wrapt==1.17.2 xgboost==3.0.1
```

### Models and Techniques

* **Custom CNN** trained from scratch on dermoscopic images.
* **EfficientNetB0** with transfer learning to leverage pre-trained weights.
* **Vision Transformer (ViT)** for patch-based self-attention on images.
* **Ensemble of XGBoost, LightGBM, and CatBoost** models trained on structured metadata.

### Data

* **Source**: International Skin Imaging Collaboration (ISIC) archive
* **Size**: 401,059 dermoscopic images
* **Labels**: Benign or malignant, verified by clinical assessment or histopathology
* **Metadata**: 55 structured features used in ensemble models

### Results

* **Best Overall Model**: Ensemble (F1 = 0.59)
* **Most Clinically Useful**: ViT (Recall = 0.64 for malignant cases)

### Ethical Considerations

We account for overfitting on augmented images, dataset bias, and the real-world implications of false negatives in medical diagnosis models.

---

For more details, visit the repository: [GitHub Repo](https://github.com/Lasbaskanenmas)
