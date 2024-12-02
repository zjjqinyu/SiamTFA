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

Then set the paths of the project and dataset in "SiamTFA/ltr/admin/local.py" and "SiamTFA/pytracking/evaluation/local.py".

## Training
Set teh training parameters in  "SiamTFA/ltr/train_settings/siamtfa/siamtfa_tracker_settings.py".

Then run:
```
python SiamTFA/ltr/run_training.py
```

## Testing
Set the model weight path in "SiamTFA/pytracing/parameter/siamtfa/siamtfa.py".

Set the dataset to be evaluated in in "SiamTFA/pytracking/run_tracker.py".

Then run:
```
python SiamTFA/pytracking/run_tracker.py
```
