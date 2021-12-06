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
            from input import gen_wp_input, write_input
            for W in W_vals:
                for B in B_vals:
                    for A in A_vals:
                        for p_vec in p_vec_list:
                            input_data, filename = gen_wp_input(A,B,W,p_vec[0],p_vec[1])
                            write_input(filename,input_data)
                            with open(filename) as f:
                                for line in f:
                                    line1 = line.replace(" ", "")
                                    line1 = line1.rstrip('\n')
                                    name, value = line1.split("=")
                                    exec(str(line), globals())
                            copyfile(filename, 'inputfile.tmp')
                            print('starting import')
                            from wp import runSim, genviz
                            print('finished import')
                            if not (os.path.exists(calcdir)):
                                os.mkdir(calcdir)
                                runSim()
                                genviz()
                            else:
                                print('found')
                                genviz()
                            os.remove('inputfile.tmp')
                            del runSim
                            del genviz
                            del sys.modules['wp']


        else:
            # run wavepacket dynamics
            from wp import runSim, genviz
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
            from input import gen_sh_input, write_input
            for W in W_vals:
                for B in B_vals:
                    for A in A_vals:
                        for p_vec in p_vec_list:
                            input_data, filename = gen_sh_input(A, B, W, p_vec[0], p_vec[1])
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
                            else:
                                print('found')
                                # genviz()
                            os.remove('inputfile.tmp')
                            del runSim
                            del genviz
                            del sys.modules['fssh']
        else:
            # run FSSH dynamics
            from fssh import runSim, genviz
            if not (os.path.exists(calcdir)):
                os.mkdir(calcdir)
            runSim()
            genviz()
            #else:
                #genviz()
            os.remove('inputfile.tmp')
            sys.exit()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
