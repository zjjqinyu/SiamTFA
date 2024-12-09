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

Then run:
```
python pytracking/run_tracker.py --dataset_name rgbt234
```
## Tracking results
Download the tracking results from [Baidu Netdisk](https://pan.baidu.com/s/1n31MZ32ZNzSuYhaRsd-X5Q?pwd=pm6p) code: pm6p

Download the model weights from [Baidu Netdisk](https://pan.baidu.com/s/1koibm_DHj194yHpihfyf8Q?pwd=143t) code: 143t

## Acknowledgments
Thanks for the [PyTracking](https://github.com/visionml/pytracking) and [OSTrack](https://github.com/botaoye/OSTrack) library, which helps us to quickly implement our ideas.

