export CUDA_VISIBLE_DEVICES=1

python -u run.py --data EXC --root_path ./dataset/Customer --data_path exchange_rate.txt --model Lstm --binary 1 --mpa 2

python -u run.py --data EXC --root_path ./dataset/Customer --data_path exchange_rate.txt --model Lstm --binary 2 --mpa 2

python -u run.py --data EXC --root_path ./dataset/Customer --data_path exchange_rate.txt --model TpaLstm --binary 1 --mpa 2

python -u run.py --data ETTm1 --root_path ./dataset/ETT --data_path ETTm1.csv  --model Lstm --binary 1 --mpa 2

python -u run.py --data ETTm1 --root_path ./dataset/ETT --data_path ETTm1.csv --model Lstm --binary 2 --mpa 2

python -u run.py --data ETTm1 --root_path ./dataset/ETT --data_path ETTm1.csv --model TpaLstm --binary 1 --mpa 2