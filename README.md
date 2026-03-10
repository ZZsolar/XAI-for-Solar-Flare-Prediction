# code_work — Solar flare forecasting workflow

This directory contains the full pipeline for solar flare forecasting from SHARP magnetograms and multi-source parameters: data preparation, CNN training and evaluation, mask generation (MFR/PIL), SHARP parameter computation, and parameter-based classification.

## Directory structure

```
code_work/
├── README.md
├── data_preparation.ipynb   # Data prep: time windows, labels, train/valid/test split
├── train_model.ipynb        # CNN training
├── pipeline.ipynb           # Classification pipeline from SHARP/MFR/PIL parameters
├── analysis.ipynb           # Model evaluation, RoI and Grad-CAM analysis
├── code_mask/               # Masks and SHARP parameter computation
│   ├── MFR_mask.py          # MFR masks from Grad-CAM
│   ├── PIL_mask.py          # Polarity Inversion Line (PIL) masks
│   ├── SHARP_masked.py      # SHARP parameters under mask (ori / pil / mfr)
│   └── Calculate_sharpkeys_masked.py  # SHARP parameter function library
├── cnn/
│   └── utils/
│       ├── model.py         # CNN model definition
│       └── utils.py         # Dataset, transforms, evaluation, FITS helpers
├── data/                    # Data and splits
│   ├── split/               # train/valid/test CSV splits
│   ├── param_data/          # SHARP/MFR/PIL parameter CSVs (a2/a3 aligned)
│   └── ...
└── param/                   # Pipeline outputs: parameters and performance
```

## Workflow

1. **Data preparation**. Run `data_preparation.ipynb`: build time windows in `operational_form.csv` from flare records, assign positive/negative labels, and create train/valid/test splits in `data/split/*.csv` and `data/label.csv`.
2. **CNN training**. Run `train_model.ipynb`: train the binary CNN using `data/split/train.csv`, `valid.csv`, and the dataset/model in `cnn/utils`. Models are saved under `model/` (or your configured path).
3. **Mask generation**

   - **MFR**: run `code_mask/MFR_mask.py` to produce MFR attention masks (`.attr.npy`) from the trained CNN and Grad-CAM.
   - **PIL**: run `code_mask/PIL_mask.py` to compute the polarity inversion line from Br.fits and save PIL masks (`.pil.npy`).
4. **SHARP parameters (masked)**. Run `code_mask/SHARP_masked.py` with `--method ori`, `pil`, or `mfr`. SHARP parameters are computed under the chosen mask and written to the corresponding CSV under `param_data`.
5. **Parameter pipeline**. Run `pipeline.ipynb`: load SHARP/MFR/PIL parameters, clean and align (a2: SHARP+MFR; a3: SHARP+MFR+PIL), train/evaluate SVM, Random Forest, etc., and write results to `param/`.
6. **Analysis and visualization**. Run `analysis.ipynb`: analysis in this work, such as: predictive capability evaluation, magnetic complexity analysis, and so on.

## Dependencies

- Python 3.10
- Scientific stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `astropy`, `torch`, `torchvision`
- Optional: `captum` (Grad-CAM), `sunpy` (PIL-related), `skimage`

## References and acknowledgements

- **SHARP parameter code** is derived from [mbobra/SHARPs](https://github.com/mbobra/SHARPs).
- **PIL (polarity inversion line) computation** is derived from [RanHao1999/Flare_SHARP](https://github.com/RanHao1999/Flare_SHARP). We modified it with vectorized computation.
- **MFR masks** are produced in-house using CNN Grad-CAM on the trained flare prediction model.
- **SHARP/HMI data** are from the [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/).
