
dos2unix ./data/celeba.bak/list_attr_celeba.txt 
dos2unix ./data/celeba.bak/list_attr_celeba_20pc.txt
awk 'NR<=2002{if(NR>2){print $0,"0",NR%16}else{print $0}}' ./data/celeba.bak/list_attr_celeba.txt > ./data/celeba/list_attr_celeba.txt
awk 'NR<=2000' ./data/celeba.bak/list_eval_partition.txt > ./data/celeba/list_eval_partition.txt
awk 'BEGIN{a=0}{if($1~/002/){a=1}if(a==0){print $0}}' ./data/celeba.bak/list_attr_celeba_20pc.txt > ./data/celeba/list_attr_celeba_20pc.txt
n=$(wc -l ./data/celeba/list_attr_celeba_20pc.txt | cut -f 1 -d ' '  )
n=$((n-2))
sed -ie "1s/.*/$n/" ./data/celeba/list_attr_celeba_20pc.txt 
n=$(wc -l ./data/celeba/list_attr_celeba.txt | cut -f 1 -d ' '  )
n=$((n-2))
sed -ie "1s/.*/$n/" ./data/celeba/list_attr_celeba.txt

vi /home/chenzhao/project/facegenerator_matchgan/MatchGAN/data_loader.py
        lines_to_shuffle = lines[:1600]
        random.seed(1234)
        random.shuffle(lines_to_shuffle)
        lines = lines_to_shuffle + lines[1600:]

python main.py --mode eval --image_size 128 --c_dim 5 \
                   --dataset CelebA \
                   --celeba_image_dir ./data/celeba/images/ \
                   --attr_path ./data/celeba/list_attr_celeba.txt \
                   --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                   --model_save_dir matchgan_celeba/models \
                   --test_iters 200000 \
                   --device 0 --num_iters 200000 \
                   --log_dir matchgan_celeba/results/logs_eval \
                   --result_dir matchgan_celeba/results_eval


python main.py --mode $mode --dataset CelebA --image_size 128 --c_dim 5 \
                   --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                   --model_save_dir matchgan_celeba/models \
                   --result_dir matchgan_celeba/results \
                   --log_dir matchgan_celeba/results/logs \
                   --attr_path ./data/celeba/list_attr_celeba_${labelled_percentage}pc.txt \
                   --labelled_percentage $labelled_percentage \
                   --device $device --num_iters $numi
				   

attr_path="./data/celeba/list_attr_celeba.txt"
lines = [line.rstrip() for line in open(attr_path, 'r')]

attr2idx = {}
idx2attr = {}
all_attr_names = lines[1].split()
for i, attr_name in enumerate(all_attr_names):
    attr2idx[attr_name] = i
    idx2attr[i] = attr_name
	
lines_to_shuffle = lines[:1600]
random.seed(1234)
random.shuffle(lines_to_shuffle)
lines = lines_to_shuffle + lines[1600:]

for i, line in enumerate(lines):
    split = line.split()
    filename = split[0]
    values = split[1:41]
    flag = int(split[41])



import os
import argparse
from solver import Solver
from subsample import subsample
from data_loader import get_loader
from torch.backends import cudnn
from itertools import product
from train_on_fake import train_on_fake
from gan_test import test_train_on_fake

celeba_image_dir="./data/celeba/images/"
attr_path="./data/celeba/list_attr_celeba.txt"
selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young"
selected_attrs=selected_attrs.split()
celeba_crop_size=178
image_size=128
batch_size=50
mode="eval"
num_workers=1

def generate_sensible_labels(selected_attrs):
	hair_color_indices = []
	for i, attr_name in enumerate(selected_attrs):
		if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
			hair_color_indices.append(i)
			
	labels_dict = {}
	label_id = 0
	for c_trg in product(*[[-1, 1]] * len(selected_attrs)):
		hair_color_sublabel = [c_trg[i] for i in hair_color_indices]
		if sum(hair_color_sublabel) > -1:
			continue
		else:
			labels_dict[label_id] = list(c_trg)
			label_id += 1
	return labels_dict
	

labels_dict = generate_sensible_labels(selected_attrs)

labelled_loader = get_loader(celeba_image_dir,
                                     attr_path,
                                     selected_attrs,
                                     False,
                                     celeba_crop_size,
                                     image_size,
                                     batch_size,
                                     'CelebA',
                                     'train_all',
                                     num_workers)
	
vi main.py
# else: ...
labelled_loader = get_loader(celeba_image_dir,
                                     attr_path,
                                     selected_attrs,
                                     False,
                                     celeba_crop_size,
                                     image_size,
                                     100,
                                     'Custom',
                                     'train_all',
                                     num_workers)



vi solver.py
print(out_cls," ",label_org)



