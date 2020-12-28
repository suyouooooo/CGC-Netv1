import os
import sys
import time

# cmd_0 = 'nohup python -u train.py --alpha 1.0 --batch-size 8 --lr 0.0005 --epochs 1000 --gpu 0 --save-dir ./experiments/GlaS/1  > Glas_1.log 2>&1 &'
# cmd_1 = 'nohup python -u train.py --alpha 1.0 --batch-size 8 --lr 0.0005 --epochs 1000 --gpu 1 --save-dir ./experiments/GlaS/1  > Glas_1.log 2>&1 &'

cmd_0 = 'nohup sh parallel_train.sh > train_output_file.txt 2>&1 &'
cmd_1 = 'nohup sh parallel_train.sh > train_output_file.txt 2>&1 &'



def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    # print(type(gpu_status))
    # print(gpu_status)
    gpu_0_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_0_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    gpu_1_memory = int(gpu_status[6].split('/')[0].split('M')[0].strip())
    gpu_1_power = int(gpu_status[5].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_0_power, gpu_0_memory, gpu_1_power, gpu_1_memory


def narrow_setup(interval=2):
    gpu_0_power, gpu_0_memory, gpu_1_power, gpu_1_memory = gpu_info()
    i = 0

    while gpu_0_memory > 1 and gpu_1_memory > 1 :  # set waiting condition
        gpu_0_power, gpu_0_memory, gpu_1_power, gpu_1_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_0_power_str = 'gpu 0 power:%d W |' % gpu_0_power
        gpu_0_memory_str = 'gpu 0  memory:%d MiB |' % gpu_0_memory
        gpu_1_power_str = 'gpu 1 power:%d W |' % gpu_1_power
        gpu_1_memory_str = 'gpu 1 memory:%d MiB |' % gpu_1_memory
        # sys.stdout.write('\r' + gpu_0_memory_str + ' ' + gpu_0_power_str + ' ' + gpu_1_memory_str + ' ' + gpu_1_power_str + ' ' + symbol)
        # sys.stdout.write('\r' + gpu_1_memory_str + ' ' + gpu_1_power_str + ' ' + symbol)
        # sys.stdout.flush()
        time.sleep(interval)
        i += 1
    if gpu_0_memory <= 1 and gpu_1_memory > 1:
        print('\n' + cmd_0)
        os.system(cmd_0)
    elif gpu_1_memory <= 1 and gpu_0_memory > 1:
        print('\n' + cmd_1)
        os.system(cmd_1)
    else:
        print('\n' + cmd_0)
        os.system(cmd_0)


if __name__ == '__main__':
    narrow_setup()