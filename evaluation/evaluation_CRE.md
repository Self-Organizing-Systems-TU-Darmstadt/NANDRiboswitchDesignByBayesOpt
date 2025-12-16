# CRE Specifics
https://github.com/sjgosai/boda2?tab=readme-ov-file#overview

## Setup
Boda2 respectively Malinois repository setup requires Python 3.11 or below as there are dependency issues with >=3.12. 

## Training Malinois
The training of Malinois defines train, validation and test set via chromosome numbers. To adapt this to smaller sizes, the respective train, test and validation datasets have to be merged into a single file with the chromosome number doing the differentiation.

Training command according to their GitHub
```
python /home/ubuntu/boda2/src/train.py \
  --data_module=MPRA_DataModule \
    --datafile_path=gs://tewhey-public-data/CODA_resources/Table_S2__MPRA_dataset.txt \
    --sep tab --sequence_column sequence \
    --activity_columns K562_log2FC HepG2_log2FC SKNSH_log2FC \
    --stderr_columns K562_lfcSE HepG2_lfcSE SKNSH_lfcSE \
    --stderr_threshold 1.0 --batch_size=1076 \
    --duplication_cutoff=0.5 --std_multiple_cut=6.0 \
    --val_chrs 7 13 --test_chrs 9 21 X \
    --synth_val_pct=0.0 --synth_test_pct=99.98 \
    --padded_seq_len=600 --use_reverse_complements=True --num_workers=8 \
  --model_module=BassetBranched \
    --input_len 600 \
    --conv1_channels=300 --conv1_kernel_size=19 \
    --conv2_channels=200 --conv2_kernel_size=11 \
    --conv3_channels=200 --conv3_kernel_size=7 \
    --linear_activation=ReLU --linear_channels=1000 \
    --linear_dropout_p=0.11625456877954289 \
    --branched_activation=ReLU --branched_channels=140 \
    --branched_dropout_p=0.5757068086404574 \
    --n_outputs=3 --n_linear_layers=1 \
    --n_branched_layers=3 --n_branched_layers=3 \
    --use_batch_norm=True --use_weight_norm=False \
    --loss_criterion=L1KLmixed --beta=5.0 \
    --reduction=mean \
  --graph_module=CNNTransferLearning \
    --parent_weights=gs://tewhey-public-data/CODA_resources/my-model.epoch_5-step_19885.pkl \
    --frozen_epochs=0 \
    --optimizer=Adam --amsgrad=True \
    --lr=0.0032658700881052086 --eps=1e-08 --weight_decay=0.0003438210249762151 \
    --beta1=0.8661062881299633 --beta2=0.879223105336538 \
    --scheduler=CosineAnnealingWarmRestarts --scheduler_interval=step \
    --T_0=4096 --T_mult=1 --eta_min=0.0 --last_epoch=-1 \
    --checkpoint_monitor=entropy_spearman --stopping_mode=max \
    --stopping_patience=30 --accelerator=gpu --devices=1 --min_epochs=60 --max_epochs=200 \
    --precision=16 --default_root_dir=/tmp/output/artifacts \
    --artifact_path=gs://your/bucket/mpra_model/
```