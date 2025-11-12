export WORLD_SIZE=2
export RENDEZVOUS=env://
export MASTER_ADDR= #IP address of master
export MASTER_PORT=
export RANK=1
export GLOO_SOCKET_IFNAME=
export OMP_NUM_THREADS=$(nproc)
export KMP_AFFINITY=granularity=fine,compact,1,0

