export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./bCapsule

cd bert_pytorch
python setup.py install
cd ..
python run_classifier.py
#python run_classifier.py --data_dir=/home/amit/bCapsule/data --task_name=bCapsule \
#    --do_train \
#	--do_eval \
#    --bert_model=./pre_trained \
#    --max_seq_length=100 \
#	--Nways=10 \
#	--Kshot=5 \
#	--meta_steps=64 \
#	--meta_batch=16 \
#    --lr_a=5e-4 \
#	--learning_rate=5e-5 \
#    --test_train_epochs=8 \
#    --output_dir=./output
