# The working directory is `mim-example/mmcls_custom_backbone`
# Training
PYTHONPATH=$PWD:$PYTHONPATH mim train mmedit ./config/SCETx2.py --gpus 1 --work-dir MyExperiment
