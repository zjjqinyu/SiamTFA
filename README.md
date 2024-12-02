# SiamTFA: Siamese Triple-stream Feature Aggregation Network for Efficient RGBT Tracking
The paper was accepted by the IEEE Transactions on Intelligent Transportation Systems.

## Install the environment
Install virtual environment and dependency packages.
```bash
conda create -n siamtfa python=3.7
conda activate siamtfa
pip install -r requirements.txt
```

Create the default environment setting files.
```
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

Then set the paths of the project and dataset in "ltr/admin/local.py" and "pytracking/evaluation/local.py".

## Training
Set the training parameters in  "ltr/train_settings/siamtfa/siamtfa_tracker_settings.py".

Then run:
```
python ltr/run_training.py
```

## Testing
Set the model weight path in "pytracing/parameter/siamtfa/siamtfa.py".

Set the dataset to be evaluated in in "pytracking/run_tracker.py".

Then run:
```
python pytracking/run_tracker.py
```
