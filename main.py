import sys
import os
from shutil import copyfile
import numpy as np

def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: python main.py inputfile')
    inputfile = args[0]
    with open(inputfile) as f:
        for line in f:
            line1 = line.replace(" ", "")
            line1 = line1.rstrip('\n')
            name, value = line1.split("=")
            exec(str(line), globals())
    copyfile(inputfile, 'inputfile.tmp')
    if 'scan' in vars() or 'scan' in globals():
        print('Running Scan...')
        scan=True
    else:
        scan=False
    if sim == 'WP':
        if scan:
            if model == 1:
                from input import gen_wp_input, write_input
                for W in W_vals:
                    for B in B_vals:
                        for A in A_vals:
                            for p_vec in p_vec_list:
                                input_data, filename = gen_wp_input(A, B, W, p_vec[0], p_vec[1], xran, yran, pxmax,
                                                                    pymax)
                                write_input(filename, input_data)
                                with open(filename) as f:
                                    for line in f:
                                        line1 = line.replace(" ", "")
                                        line1 = line1.rstrip('\n')
                                        name, value = line1.split("=")
                                        exec(str(line), globals())
                                copyfile(filename, 'inputfile.tmp')
                                from wp import runSim, genviz
                                if not (os.path.exists(calcdir)):
                                    os.mkdir(calcdir)
                                    runSim()
                                    genviz()
                                else:
                                    genviz()
                                os.remove('inputfile.tmp')
                                del runSim
                                del genviz
                                del sys.modules['wp']
            if model == 2:
                from input import gen_wp_input_2, write_input
                for W in W_vals:
                    for B in B_vals:
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_2(alpha, A, B, W, p_vec[0], p_vec[1], xran,
                                                                          yran, pxmax, pymax)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, 'inputfile.tmp')
                                    from wp_2 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove('inputfile.tmp')
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_2']
            if model==3:
                from input import gen_wp_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3(alpha, A,B,W,p_vec[0],p_vec[1],xran,yran,pxmax,pymax)
                                    write_input(filename,input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, 'inputfile.tmp')
                                    from wp_2 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove('inputfile.tmp')
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_3']


        else:
            # run wavepacket dynamics
            from wp_2 import runSim, genviz
            if not(os.path.exists(calcdir)):
                os.mkdir(calcdir)
                runSim()
                genviz()
            else:
                genviz()
            os.remove('inputfile.tmp')
            sys.exit()
    if sim == 'MF':
        # run mean-field dynamics
        from mf import runSim, genviz
        if not (os.path.exists(calcdir)):
            os.mkdir(calcdir)
        runSim()
        genviz()
                #genviz()
            #else:
            #    break
                #genviz()
        os.remove('inputfile.tmp')
        sys.exit()
    if sim == 'FSSH':
        if scan:
            if model==1:
                from input import gen_sh_input, write_input
                for W in W_vals:
                    for B in B_vals:
                        for A in A_vals:
                            for p_vec in p_vec_list:
                                input_data, filename = gen_sh_input(A, B, W, N, p_vec[0], p_vec[1], rescale_method)
                                write_input(filename, input_data)
                                with open(filename) as f:
                                    for line in f:
                                        line1 = line.replace(" ", "")
                                        line1 = line1.rstrip('\n')
                                        name, value = line1.split("=")
                                        exec(str(line), globals())
                                copyfile(filename, 'inputfile.tmp')
                                from fssh import runSim, genviz
                                if not (os.path.exists(calcdir)):
                                    os.mkdir(calcdir)
                                runSim()
                                genviz()
                                os.remove('inputfile.tmp')
                                del runSim
                                del genviz
                                del sys.modules['fssh']
            if model == 2:
                from input import gen_sh_input_2, write_input
                for W in W_vals:
                    for B in B_vals:
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_sh_input_2(alpha, A, B, W, N, p_vec[0], p_vec[1],
                                                                          r_init[0], r_init[1], rescale_method)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, 'inputfile.tmp')
                                    from fssh_2 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    runSim()
                                    genviz()
                                    os.remove('inputfile.tmp')
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_2']
            if model==3:
                from input import gen_sh_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_sh_input_3(alpha,A, B, W, N, p_vec[0], p_vec[1],r_init[0],r_init[1], rescale_method)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, 'inputfile.tmp')
                                    from fssh_3 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    runSim()
                                    genviz()
                                    os.remove('inputfile.tmp')
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_3']
        else:
            # run FSSH dynamics
            from fssh import runSim, genviz
            if not (os.path.exists(calcdir)):
                os.mkdir(calcdir)
            runSim()
            genviz()
            os.remove('inputfile.tmp')
            sys.exit()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
