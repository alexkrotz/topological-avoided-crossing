import sys
import os
from shutil import copyfile
import glob
import numpy as np


def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: python main.py inputfile')
    inputfile = args[0]
    num_tmpfiles = len(glob.glob('inputfile.tmp-*')) + 1
    tmpfile = 'inputfile.tmp-' + str(num_tmpfiles)
    viz_only = False
    with open(inputfile) as f:
        for line in f:
            line1 = line.replace(" ", "")
            line1 = line1.rstrip('\n')
            name, value = line1.split("=")
            exec(str(line), globals())
    copyfile(inputfile, tmpfile)
    print('Running Scan...')
    scan=True
    if sim == 'WP':
        from input import gen_wp_input, write_input
        for W in W_vals:
            for B in B_vals:
                Bx = B[0]
                By = B[1]
                for A in A_vals:
                    for alpha in alpha_vals:
                        for p_vec in p_vec_list:
                            input_data, filename = gen_wp_input(alpha, A,Bx,By,W,p_vec[0],p_vec[1],xran,yran,\
                                r_init,pxmax,pymax,dtfac=dt_fac,t_mult=t_mult,init_diab=init_diab,model=model)
                            write_input(filename,input_data)
                            with open(filename) as f:
                                for line in f:
                                    line1 = line.replace(" ", "")
                                    line1 = line1.rstrip('\n')
                                    name, value = line1.split("=")
                                    exec(str(line), globals())
                            copyfile(filename, tmpfile)
                            from wp import runSim, genviz
                            if not (os.path.exists(calcdir)):
                                os.mkdir(calcdir)
                                runSim()
                                genviz()
                            else:
                                genviz()
                            os.remove(tmpfile)
                            del runSim
                            del genviz
    if sim == 'FSSH':
        from input import gen_sh_input, write_input
        for W in W_vals:
            for B in B_vals:
                Bx = B[0]
                By = B[1]
                for A in A_vals:
                    for alpha in alpha_vals:
                        for p_vec in p_vec_list:
                            input_data, filename = gen_sh_input(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                  r_init[0], r_init[1], rescale_method, t_mult,
                                                                  decohere=False, init_diab=init_diab, model=model)
                            write_input(filename, input_data)
                            with open(filename) as f:
                                for line in f:
                                    line1 = line.replace(" ", "")
                                    line1 = line1.rstrip('\n')
                                    name, value = line1.split("=")
                                    exec(str(line), globals())
                            copyfile(filename, tmpfile)
                            from fssh import runSim, genviz
                            if not (os.path.exists(calcdir)):
                                os.mkdir(calcdir)
                            if not (viz_only):
                                runSim()
                            genviz()
                            os.remove(tmpfile)
                            del runSim
                            del genviz
                            del sys.modules['fssh']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
