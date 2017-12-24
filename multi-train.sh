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

#train_data=train_textbook.txt
#valid_data=valid_textbook.txt
#test_data=test_textbook.txt

#train_data=train_twit.txt
#valid_data=valid_twitter.txt
#test_data=test_twitter.txt

#train_data=train_all.txt
#valid_data=valid_all.txt
#test_data=test_all.txt

echo -n "type [directory name]: "
read directory_name
echo directory_name: $directory_name


#embedding_size=300
#hidden_size=256
#dropout=0
for hidden_size in 128 256 512 1024
do
	echo hidden_size: $hidden_size
	for dropout in 0 0.1 0.2 0.5
	echo dropout: $dropout
		for try in {1..5..1}
		do
			python main.py $train_data $valid_data $test_data $directory_name $model_name $hidden_size $dropout
		done
	done
done