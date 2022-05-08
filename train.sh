# The working directory is `mim-example/mmcls_custom_backbone`
# Training
PYTHONPATH=$PWD:$PYTHONPATH mim train mmedit /home/chenz/NTIRE2022/NTIRE2022/SCET_mim/config/SCET_x3_128.py --gpus 1 --work-dir MyExperiment
