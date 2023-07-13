
# Train CNN + ML-Deocder on Kuzushiji-MNIST
python main.py --dataset k10 --num_classes 10 --img_size 28 --batch_size 200 --epoch 20 --backbone cnn --learning_rate 0.0001 --checkpoint_path ./checkpoint/cnn.pt --query_weight ./dataset/query_embedding_k10.pt --freeze_query

# Train CNN + ML-Deocder on Kuzushiji-49
python main.py --dataset k49 --num_classes 49 --img_size 28 --batch_size 200 --epoch 20 --backbone cnn --learning_rate 0.0001 --checkpoint_path ./checkpoint/cnn_k49.pt --query_weight ./dataset/query_embedding_k49.pt --freeze_query


# Train Residual CNN + ML-Deocder on Kuzushiji-MNIST
python main.py --dataset k10 --num_classes 10 --img_size 28 --batch_size 200 --epoch 20 --backbone rescnn --learning_rate 0.0001 --checkpoint_path ./checkpoint/rescnn.pt --query_weight ./dataset/query_embedding_k10.pt --freeze_query

# Train Residual CNN + ML-Deocder on Kuzushiji-49
python main.py --dataset k49 --num_classes 49 --img_size 28 --batch_size 200 --epoch 20 --backbone rescnn --learning_rate 0.0001 --checkpoint_path ./checkpoint/rescnn_k49.pt --query_weight ./dataset/query_embedding_k49.pt --freeze_query