#!/bin/sh
echo "run python crawling.py 10 times"
for iter in {1..10..1}
do
	echo crawling iteration: $iter
	python crawling.py
done
