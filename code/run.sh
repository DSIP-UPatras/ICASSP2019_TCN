#!/bin/bash
LAST_SUBJECT="27"

## Average over Time, RF:300ms
SUBJECT="1"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
while [ $SUBJECT -le $LAST_SUBJECT ]
do
	python3 script_experiment.py config/TCCNet_aot_300.json $SUBJECT $TIMESTAMP
	SUBJECT=$[$SUBJECT+1]
done

## Average over Time, RF:2500ms
SUBJECT="1"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
while [ $SUBJECT -le $LAST_SUBJECT ]
do
	python3 script_experiment.py config/TCCNet_aot_2500.json $SUBJECT $TIMESTAMP
	SUBJECT=$[$SUBJECT+1]
done

## Attention, RF:300ms
SUBJECT="1"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
while [ $SUBJECT -le $LAST_SUBJECT ]
do
	python3 script_experiment.py config/TCCNet_att_300.json $SUBJECT $TIMESTAMP
	SUBJECT=$[$SUBJECT+1]
done

## Attention, RF:2500ms
SUBJECT="1"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
while [ $SUBJECT -le $LAST_SUBJECT ]
do
	python3 script_experiment.py config/TCCNet_att_2500.json $SUBJECT $TIMESTAMP
	SUBJECT=$[$SUBJECT+1]
done
