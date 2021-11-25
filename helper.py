import os
import sys
from shutil import move

def archive_log(dst_dir = './results/', log_dir = './'):
    file_list = os.listdir(log_dir)
    # print(file_list)
    for file in file_list:
        if file.endswith('.log'):
            exp = file.split('.')[0]
            grp = exp.split('_')[0]
            dst_path = os.path.join(dst_dir, grp, exp)
            log_path = os.path.join(log_dir, exp + '.log')
            with open(log_path, 'r') as f:
                res = f.read()[-300:].split('\n')
                end_flag = ('Performance of last 5 epochs' in res)
            if sys.argv[2] == 'move':
                if os.path.isfile(log_path) and os.path.isdir(dst_path) and end_flag:
                    move(log_path, dst_path)
                    print('Exp [%s] archived.' % exp)
                else:
                    print('Exp [%s] skipped.' % exp)
            else:
                if end_flag:
                    print(dst_path)
                    print(log_path)
                    print(end_flag)

def dump_results(log_dir='./'):
    
    # fir_half = input('Please enter first half: ')
    # sec_half = '.log' # input('Please enter second half: ')
    # count = int(input('Please enter file num: '))
    # file_list = [('%s%d%s' % (fir_half, i+1, sec_half)) for i in range(count)]
    # print(file_list)

    res_miou = []
    res_pixacc = []
    res_time = []
    for file in sorted(os.listdir(log_dir)):
        if not file.endswith('.log'):
            continue
        print(os.path.basename(file))
        with open(file, 'r') as f:
            res = f.read()
            if len(res) < 310:
                print('Skipped [%s] since it\'s too short.' % file)
                res_miou.append('TBD')
                res_pixacc.append('TBD')
                res_time.append('TBD')
                continue
            res = res[-300:].split('\n')
            # print(res)
            if 'Performance of last 5 epochs' in res:
                idx = res.index('Performance of last 5 epochs')
                # final_idx = res.index('Performance of last 5 epochs')
                res_miou1 = eval(res[idx+1].split(': ')[-1])
                res_pixacc1 = eval(res[idx+2].split(': ')[-1])
                res_miou2, res_pixacc2 = eval(res[idx+3].split(': ')[-1])
                if 'False' in res[idx+5]:
                    res_t = res[idx+6].split(': ')[-1]
                else:
                    res_t = res[idx+5].split(': ')[-1]
                # print(res_miou1, res_miou2)
                # print(res_pixacc1, res_pixacc1)
                res_miou.append('%.4f / %.4f' % (res_miou1, res_miou2))
                res_pixacc.append('%.4f / %.4f' % (res_pixacc1, res_pixacc2))
                res_time.append('%d epochs / %s' % (600, res_t))
            else:
                print('Skipped [%s] since it\'s incomplete.' % file)
                res_miou.append('TBD')
                res_pixacc.append('TBD')
                res_time.append('TBD')
    
    print('[mIoU]:', '\n'.join(res_miou), sep='\n')
    print('\n[Pix_Acc]:', '\n'.join(res_pixacc), sep='\n')
    print('\n[Time]:', '\n'.join(res_time), sep='\n')

if __name__ == '__main__':
    # print(sys.argv)
    if sys.argv[1] == 'log':
        archive_log()
    elif sys.argv[1] == 'dump':
        dump_results()