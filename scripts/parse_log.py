import argparse
import sys
from collections import defaultdict
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_log', help='input log file', required=True)
    parser.add_argument('-r','--range', nargs='+', help='average over epoch range', default=[0,10], type=int)
    parser.add_argument('-k','--keys', nargs='+', help='keys to parse  in the log file', default=['dice_score='])
    args = parser.parse_args()
    return vars(args)


def parse_log(log_file, keys, seperator=" "):
    dict_val = defaultdict(list)
    # import ipdb as pdb; pdb.set_trace()
    with open(log_file) as f:
        logs = f.readlines()
        for log in logs:
            for key in keys:
                if key in log:
                    log = ' '.join(log.split())
                    val = float(log.split(key)[1].split(seperator)[0][:-1])
                    dict_val[key].append(val)
    return dict_val

def report_val(keys, dict, epoch_range):
    max = epoch_range[1]
    min = epoch_range[0]
    #import ipdb as pdb; pdb.set_trace()
    if min<0 or max<0 :
        print("specified range [{}:{}] is not valid".format(min, max))
        sys.exit()
    for key in keys:
        arry_len = len(np.asarray(dict[key]))
        if max>arry_len or min>arry_len:
            print("specified range [{}:{}] is out of valid range [0:{}]".format(min, max, arry_len))
            sys.exit()
        vals = np.asarray(dict[key])[min:max]
        print("{:>15}[{:4d}:{:4d}]: average={:2.4f}    min={:2.4f}    max={:2.4f}".format(key, min, max, vals.mean(), vals.min(), vals.max()))


if __name__ == '__main__':
    args        = parse_args()
    input_log   = args['input_log']
    epoch_range = args['range']
    keys        = args['keys']
    val_dict = parse_log(input_log, keys, " ")
    report_val(keys, val_dict, epoch_range)
