import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import argparse
# plt.rcParams["font.family"] = "Times New Roman"


# #files = ['fullprecision.txt', "a8_8_w8_8.txt","a4_0_w0_4.txt","a4_2_w0_4.txt","a4_0_w0_2.txt"]
# files      = ["log_2_bit_QReg.txt", "log_4_bit_QReg.txt", "log_6_bit_QReg.txt", "log_8_bit_QReg.txt", "log_src_QReg.txt"]
# #files      = ["log_6_bit_QReg.txt", "log_8_bit_QReg.txt", "log_src_QReg.txt"]
# line_style = ["-.", "", "-.", "", "-.", "", "-.", "" ]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_log_dir', help='path to log file folder', required=False, default="/export/tmp/hemmat/results/")
    parser.add_argument('-s', '--line_styles', nargs='+', help='line styles', required=True)
    parser.add_argument('-l','--log_files', nargs='+', help='log files', required=True)
    args = parser.parse_args()
    return vars(args)

# Using smooth function to smooth the data
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot(files, line_style):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for idx, log_file in enumerate(files):
        scores = []
        with open(log_file, "r") as f:
            lines = f.readlines()
            for num, line in enumerate(lines):
                if "dice_score" in line.lower():
                    score = float(line.split("dice_score=")[-1].split("\n")[0][:-1])
                    scores.append(score)
                else:
                    pass
        scores_smooth = smooth(scores, 20)
        scores_smooth[-20:] = scores[-20:]
        # import ipdb as pdb; pdb.set_trace()
        if line_style[idx] == "":
            ls = "-"
        if line_style[idx] == "d-":
            ls = "-."
        else:
            ls = line_style[idx]
        plt.plot(scores_smooth, ls=ls, label=log_file.split(".txt")[0].split("/")[-1])
        plt.legend(loc='center right', fontsize='small')
    plt.xlabel('Epoch', fontsize='large')
    plt.ylabel('Accuracy', fontsize='large')
    # Major ticks every 20, minor ticks every 5
    xmajor_ticks = np.arange(0, 200, 20)
    xminor_ticks = np.arange(0, 200, 5)
    ymajor_ticks = np.arange(0, 60, 5)
    yminor_ticks = np.arange(0, 60, 1)
    ax.set_xticks(xmajor_ticks)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(ymajor_ticks)
    ax.set_yticks(yminor_ticks, minor=True)
    # And a corresponding grid
    ax.grid()
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.savefig("accuracy" + ".pdf")

if __name__ == '__main__':
    # import ipdb as pdb; pdb.set_trace()
    args = parse_args()
    line_styles = args['line_styles']
    log_files   = args['log_files']
    base_log_dir= args['base_log_dir']
    log_files   = [base_log_dir+log_file+"/"+log_file+".txt" for log_file in log_files]
    if len(line_styles) != len(log_files):
        import sys
        print("Number of line styles does not match log files")
        sys.exit()
    plot(log_files, line_styles)
