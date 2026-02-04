python -m torch.distributed.launch --nproc_per_node 1 --master_port 11133 tools/test.py --launcher pytorch

/home/robosense/nas/Geely-a/bag/20250118/E371_3484_20250118_1500/2025-01-18-15-02-04

/home/robosense/jiaxin/sutengOuput

python -m torch.distributed.launch --nproc_per_node 1 --master_port 11133 tools/test.py --InputPath /home/robosense/nas/Geely-a/bag/20250118/E371_3484_20250118_1500/2025-01-18-15-02-04 --launcher pytorch








conda activate open-mmlab-ealss
./run.sh /home/robosense/nas/Geely-a/bag/20250118/E371_3484_20250118_1500/2025-01-18-15-02-04



# RuntimeError: Address already in use

ps -aux | grep python

kill -9 #2406263