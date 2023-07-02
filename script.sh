python main.py --dataset k10 --num_classes 10 --img_size 64 --backbone alexnet --checkpoint_path ./checkpoint/alexnet/k10_backbone_v1.pt --train_backbone

python main.py --dataset k49 --num_classes 49 --img_size 64 --backbone alexnet --checkpoint_path ./checkpoint/alexnet/k49_backbone_v1.pt --train_backbone --learning_rate 0.005 --lr_decay 0.9 --batch_size 128

python main.py --dataset k49 --num_classes 49 --img_size 64 --backbone resnet --checkpoint_path ./checkpoint/resnet/k49_backbone_v1.pt --train_backbone --learning_rate 0.005 --lr_decay 0.9 --batch_size 128