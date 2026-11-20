# Re-training & Uploading Models on Keras 3

This guide walks through verifying the modernized deepBlink on a GPU machine,
re-training all 9 pre-trained models, and uploading the new `.h5` weights to
replace the Figshare-hosted ones.

---

## 1. Environment Setup

```bash
# Clone and switch to the modernization branch
git clone https://github.com/BBQuercus/deepBlink.git
cd deepBlink
git checkout modernization

# Create a fresh venv
uv venv .venv --python 3.12
source .venv/bin/activate

# Install deepBlink + dev deps
uv pip install -e ".[dev,wandb]"

# GPU support (if not already bundled with your TF install)
uv pip install 'tensorflow[and-cuda]'

# Verify GPU is visible
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 2. Run the Test Suite

This is the first validation checkpoint — make sure the migration didn't
break anything before spending time on training.

```bash
pytest tests/ -v
```

**Known issue:** `tests/test_cli.py` loads `tests/data/model.h5`, which was
saved with old Keras. If it fails to load, that's expected and confirms we
need to re-save models. You can skip it for now:

```bash
pytest tests/ -v --ignore=tests/test_cli.py
```

---

## 3. Verify Old Models Can Still Load

Try loading each of the 9 Figshare models to see if Keras 3 can read them.
This determines whether we need a full re-train or just a re-save.

```bash
# Download all existing models
deepblink download --all

# Test loading each one
python -c "
import deepblink
import glob

for path in sorted(glob.glob('*.h5')):
    try:
        model = deepblink.load_model(path)
        print(f'OK   {path}  layers={len(model.layers)}')
    except Exception as e:
        print(f'FAIL {path}  {e}')
"
```

**If all models load:** You can re-save them in the new format without
re-training (jump to step 5). **If any fail:** Those models need a full
re-train (continue to step 4).

---

## 4. Re-train Models

Each model corresponds to a specific dataset and config. The 9 models are:

| Model | Spot type | Dataset source |
|-------|-----------|----------------|
| `vesicle` | Vesicles | Internal |
| `receptor` | Receptors | Internal |
| `microtubule` | Microtubule tips | Internal |
| `particle` | Particles | Internal |
| `smfish1` | smFISH | Internal |
| `suntag1` | SunTag | Internal |
| `suntag2` | SunTag (v2) | Internal |
| `ms2` | MS2 | Internal |
| `smfish2` | smFISH (v2) | Internal |

### 4a. Generate a config template

```bash
deepblink config --name retrain_config --raw
```

This creates `retrain_config.yaml`. Edit it for each model — the key fields:

```yaml
name: deepBlink
run_name: vesicle  # change per model
savedir: ./retrained_models
use_wandb: false  # or true if you want logging

dataset: SpotsDataset
dataset_args:
  name: PATH/TO/DATASET.NPZ  # point to the original dataset
  cell_size: 4
  smooth_factor: 1

model: SpotsModel
network: unet
network_args:
  dropout: 0.3
  filters: 5
  ndown: 2
  l2: 1.0e-06
  block: convolutional

loss: combined_dice_rmse
optimizer: amsgrad

train_args:
  batch_size: 2
  epochs: 200
  learning_rate: 1.0e-04
  overfit: false
  pre_train: null
```

### 4b. Train each model

```bash
mkdir -p retrained_models

# For each model, update the config and run:
deepblink train -c retrain_config.yaml --gpu 0
```

Or via the Python API:

```python
import yaml
import deepblink

with open("retrain_config.yaml") as f:
    cfg = yaml.safe_load(f)

# Override per model
cfg["run_name"] = "vesicle"
cfg["dataset_args"]["name"] = "/path/to/vesicle_dataset.npz"
cfg["savedir"] = "./retrained_models"

deepblink.train(cfg)
```

### 4c. Validate the re-trained models

For each re-trained model, compare against the original:

```python
import deepblink
import numpy as np

model = deepblink.load_model("retrained_models/vesicle.h5")
image = deepblink.load_image("test_image.tif")
coords = deepblink.predict(image, model, probability=0.5)

print(f"Detected {len(coords)} spots")
# Visually inspect or compare metrics against known ground truth
```

---

## 5. Re-save Models (if old ones loaded but need format update)

If old models loaded fine in step 3 but you want to ensure they're saved
in the current Keras 3 format:

```python
import deepblink
import glob

for path in sorted(glob.glob("*.h5")):
    model = deepblink.load_model(path)
    # Re-save in current format
    model.save(f"resaved_{path}")
    print(f"Re-saved {path}")
```

---

## 6. Update test model

The test suite ships with `tests/data/model.h5`. Re-save it so CI passes:

```python
import deepblink

model = deepblink.load_model("tests/data/model.h5")
model.save("tests/data/model.h5")
```

If it can't load, build a minimal model and save it:

```python
import keras
import numpy as np
from deepblink.networks import unet

model = unet(cell_size=4, ndown=2, filters=4)
model.compile(
    optimizer="adam",
    loss=deepblink.losses.combined_dice_rmse,
    metrics=[deepblink.losses.f1_score, deepblink.losses.rmse],
)

# Quick overfit on random data so weights aren't all zeros
x = np.random.rand(4, 64, 64, 1).astype(np.float32)
y = np.random.rand(4, 16, 16, 3).astype(np.float32)
model.fit(x, y, epochs=1, verbose=0)

model.save("tests/data/model.h5")
```

Commit the updated test model:
```bash
git add tests/data/model.h5
git commit -m "Update test model for Keras 3 compatibility"
git push origin modernization
```

---

## 7. Upload Re-trained Models

### Current hosting: Figshare

The 9 models are currently hosted on Figshare and referenced in
`deepblink/cli/models.txt`. To update:

1. Upload each new `.h5` file to Figshare (same project, new version or new files)
2. Get the new download URLs
3. Update `deepblink/cli/models.txt` with the new URLs:

```python
import json

models = {
    "vesicle": "https://ndownloader.figshare.com/files/NEW_ID",
    "receptor": "https://ndownloader.figshare.com/files/NEW_ID",
    # ... all 9 models
}

with open("deepblink/cli/models.txt", "w") as f:
    json.dump(models, f)
```

4. Commit and push:
```bash
git add deepblink/cli/models.txt
git commit -m "Update pre-trained model URLs for Keras 3 re-trained weights"
git push origin modernization
```

---

## 8. Final Validation Checklist

Run through this on the GPU machine before merging:

- [ ] `pytest tests/ -v` passes (including test_cli.py with updated model)
- [ ] `deepblink download --list` shows all 9 models
- [ ] `deepblink download --input vesicle` downloads and the file loads
- [ ] `deepblink predict -m vesicle.h5 -i test_image.tif` produces output
- [ ] Python API workflow works end-to-end:
  ```python
  import deepblink
  model = deepblink.load_model("vesicle.h5")
  image = deepblink.load_image("test_image.tif")
  coords = deepblink.predict(image, model)
  print(f"{len(coords)} spots detected")
  ```
- [ ] Training from scratch completes (at least 5 epochs, one model)
- [ ] Re-trained model produces comparable F1 scores to the originals
