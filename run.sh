# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
# python train_supervised.py --trial pretrain --model_path ./checkpoints --tb_path ./tensorboardlogs --data_root ./data/

# distillation
# setting '-a 1.0' should give simimlar performance
# python train_distillation.py -r 0.5 -a 0.5 --path_t ./checkpoints/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --model_path ./dis_checkpoints --tb_path ./dis_tensorboardlogs --data_root ./data/

# evaluation
python eval_fewshot.py --model_path ./dis_checkpoints/S:resnet12_T:resnet12_miniImageNet_kd_r:0.5_a:0.5_b:0_trans_A_born1/resnet12_last.pth --data_root ./data/miniImageNet/
