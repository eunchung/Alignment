#!/bin/sh
echo -n "type [model name]: "
read model_name
echo model_name: $model_name

# echo -n "type [train data]: "
# read train_data
# echo train_data: $train_data

# echo -n "type [valid data]: "
# read valid_data
# echo valid_data: $valid_data

# echo -n "type [test data]: "
# read test_data
# echo test_data: $test_data

echo -n "tyep [directory name]: "
read directory_name
echo directory_name: $directory_name

train_data=train_acc_tw.txt
#valid_data=valid_acc_tw.txt
#test_data=test_acc_tw.txt
valid_data=test_acc_tw.txt
test_data=valid_acc_tw.txt

#embedding_size=300
#hidden_size=256
dropout=0
for hidden_size in {200..500..100}
do
	echo hidden_size: $hidden_size
	for embedding_size in {100..500..100}
	do
		#for dropout in 0 0.2 0.5
		for try in {1..5..1}
		do
			echo embedding_size: $embedding_size
			echo dropout: $dropout
			python main.py $train_data $valid_data $test_data $directory_name $model_name $embedding_size $hidden_size $dropout
		done
	done
done