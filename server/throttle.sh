#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

## replace DEV=lo with your card (e.g., eth0)
DEV=
if [ "$1" == "lan" ]
then
    sudo tc qdisc del dev $DEV root 
    ## about 10Gbps
    sudo tc qdisc add dev $DEV root handle 1: tbf rate 10gbit burst 100000 limit 10000
    ## about 10ms ping latency
    sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 10msec
    echo ".....Connection established."
    iperf3 -s -p 5201 -1
fi
