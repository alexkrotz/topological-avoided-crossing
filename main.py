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
    if 'scan' in vars() or 'scan' in globals():
        print('Running Scan...')
        scan=True
    else:
        scan=False
    if sim == 'WP3ls':
        print('Deprecated')
        exit()
        if scan:
            if model == 1:
                from input import gen_wp_input_3ls, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3ls(alpha, A, Bx, By, W, p_vec[0], p_vec[1],
                                                                            xran, yran, rinit, pxmax, pymax,
                                                                            dtfac=dt_fac, t_mult=t_mult)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_3ls_1 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_3ls_1']
            if model == 2:
                from input import gen_wp_input_3ls, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3ls(alpha, A, Bx, By, W, p_vec[0], p_vec[1],
                                                                            xran, yran, rinit, pxmax, pymax,
                                                                            dtfac=dt_fac, t_mult=t_mult)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_3ls_2 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_3ls_2']
            if model == 3:
                from input import gen_wp_input_3ls, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3ls(alpha, A, Bx, By, W, p_vec[0], p_vec[1],
                                                                            xran, yran, rinit, pxmax, pymax,
                                                                            dtfac=dt_fac, t_mult=t_mult)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_3ls_3 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_3ls_3']
    if sim == 'WP':
        if scan:
            if model == 1:
                print('Deprecated')
                exit()
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
                                del sys.modules['wp']
            if model == 2:
                print('Deprecated')
                exit()
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
                                    copyfile(filename, tmpfile)
                                    from wp_2 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_2']
            if model == 3:
                print('Deprecated')
                exit()
                from input import gen_wp_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3(alpha, A, Bx, By, W, p_vec[0], p_vec[1], xran,
                                                                          yran, r_init, pxmax, pymax, dtfac=dt_fac,
                                                                          t_mult=t_mult)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_3 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_3']
            # Model B
            if model==4:
                from input import gen_wp_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3(alpha, A,Bx,By,W,p_vec[0],p_vec[1],xran,yran,r_init,pxmax,pymax,dtfac=dt_fac,t_mult=t_mult,init_diab=init_diab)
                                    write_input(filename,input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_4 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_4']
            # Model C
            if model==5:
                from input import gen_wp_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3(alpha, A,Bx,By,W,p_vec[0],p_vec[1],xran,yran,r_init,pxmax,pymax,dtfac=dt_fac,t_mult=t_mult,init_diab=init_diab)
                                    write_input(filename,input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_5 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_5']
            # Model A
            if model==6:
                from input import gen_wp_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_wp_input_3(alpha, A, Bx, By, W, p_vec[0], p_vec[1], xran,
                                                                          yran, r_init, pxmax, pymax, dtfac=dt_fac,
                                                                          t_mult=t_mult, init_diab=init_diab)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from wp_6 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                        runSim()
                                        genviz()
                                    else:
                                        genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['wp_6']
        else:
            print('Deprecated')
            exit()
            # run wavepacket dynamics
            from wp_4 import runSim, genviz
            if not(os.path.exists(calcdir)):
                os.mkdir(calcdir)
                runSim()
                genviz()
            #else:
            #    genviz()
            #os.remove(tmpfile)
            sys.exit()
    if sim == 'MF':
        print('Deprecated')
        exit()
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
        os.remove(tmpfile)
        sys.exit()
    if sim == 'FSSH3ls':
        print('Deprecated')
        exit()
        if model == 1:
            from input import gen_sh_input_3ls, write_input
            for W in W_vals:
                for B in B_vals:
                    Bx = B[0]
                    By = B[1]
                    for A in A_vals:
                        for alpha in alpha_vals:
                            for p_vec in p_vec_list:
                                input_data, filename = gen_sh_input_3ls(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                      r_init[0], r_init[1], rescale_method, t_mult,
                                                                      decohere)
                                write_input(filename, input_data)
                                with open(filename) as f:
                                    for line in f:
                                        line1 = line.replace(" ", "")
                                        line1 = line1.rstrip('\n')
                                        name, value = line1.split("=")
                                        exec(str(line), globals())
                                copyfile(filename, tmpfile)
                                from fssh_3ls_1 import runSim, genviz
                                if not (os.path.exists(calcdir)):
                                    os.mkdir(calcdir)
                                if not (viz_only):
                                    runSim()
                                genviz()
                                os.remove(tmpfile)
                                del runSim
                                del genviz
                                del sys.modules['fssh_3ls_1']
        if model == 2:
            from input import gen_sh_input_3ls, write_input
            for W in W_vals:
                for B in B_vals:
                    Bx = B[0]
                    By = B[1]
                    for A in A_vals:
                        for alpha in alpha_vals:
                            for p_vec in p_vec_list:
                                input_data, filename = gen_sh_input_3ls(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                        r_init[0], r_init[1], rescale_method,
                                                                        t_mult,
                                                                        decohere)
                                write_input(filename, input_data)
                                with open(filename) as f:
                                    for line in f:
                                        line1 = line.replace(" ", "")
                                        line1 = line1.rstrip('\n')
                                        name, value = line1.split("=")
                                        exec(str(line), globals())
                                copyfile(filename, tmpfile)
                                from fssh_3ls_2 import runSim, genviz
                                if not (os.path.exists(calcdir)):
                                    os.mkdir(calcdir)
                                if not (viz_only):
                                    runSim()
                                genviz()
                                os.remove(tmpfile)
                                del runSim
                                del genviz
                                del sys.modules['fssh_3ls_2']
        if model==3:
            from input import gen_sh_input_3ls, write_input
            for W in W_vals:
                for B in B_vals:
                    Bx = B[0]
                    By = B[1]
                    for A in A_vals:
                        for alpha in alpha_vals:
                            for p_vec in p_vec_list:
                                input_data, filename = gen_sh_input_3ls(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                        r_init[0], r_init[1], rescale_method,
                                                                        t_mult,
                                                                        decohere)
                                write_input(filename, input_data)
                                with open(filename) as f:
                                    for line in f:
                                        line1 = line.replace(" ", "")
                                        line1 = line1.rstrip('\n')
                                        name, value = line1.split("=")
                                        exec(str(line), globals())
                                copyfile(filename, tmpfile)
                                from fssh_3ls_3 import runSim, genviz
                                if not (os.path.exists(calcdir)):
                                    os.mkdir(calcdir)
                                if not (viz_only):
                                    runSim()
                                genviz()
                                os.remove(tmpfile)
                                del runSim
                                del genviz
                                del sys.modules['fssh_3ls_3']
    if sim == 'FSSH':
        if scan:
            if model==1:
                print('Deprecated')
                exit()
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
                                copyfile(filename, tmpfile)
                                from fssh import runSim, genviz
                                if not (os.path.exists(calcdir)):
                                    os.mkdir(calcdir)
                                runSim()
                                genviz()
                                os.remove(tmpfile)
                                del runSim
                                del genviz
                                del sys.modules['fssh']
            if model == 2:
                print('Deprecated')
                exit()
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
                                    copyfile(filename, tmpfile)
                                    from fssh_2 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    runSim()
                                    genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_2']
            if model == 3:
                print('Deprecated')
                exit()
                from input import gen_sh_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_sh_input_3(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                          r_init[0], r_init[1], rescale_method, t_mult,
                                                                          decohere=False)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from fssh_3 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    if not (viz_only):
                                        runSim()
                                    genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_3']
            # MODEL B
            if model == 4:
                from input import gen_sh_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_sh_input_3(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                          r_init[0], r_init[1], rescale_method, t_mult,
                                                                          decohere=False, init_diab=init_diab)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from fssh_4 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    if not (viz_only):
                                        runSim()
                                    genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_4']
            # MODEL C
            if model==5:
                from input import gen_sh_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_sh_input_3(alpha,A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                          r_init[0],r_init[1], rescale_method,t_mult,decohere=False, init_diab=init_diab)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from fssh_5 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    if not(viz_only):
                                        runSim()
                                    genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_5']
            # MODEL A
            if model==6:
                from input import gen_sh_input_3, write_input
                for W in W_vals:
                    for B in B_vals:
                        Bx = B[0]
                        By = B[1]
                        for A in A_vals:
                            for alpha in alpha_vals:
                                for p_vec in p_vec_list:
                                    input_data, filename = gen_sh_input_3(alpha, A, Bx, By, W, N, p_vec[0], p_vec[1],
                                                                          r_init[0], r_init[1], rescale_method, t_mult,
                                                                          decohere=False, init_diab=init_diab)
                                    write_input(filename, input_data)
                                    with open(filename) as f:
                                        for line in f:
                                            line1 = line.replace(" ", "")
                                            line1 = line1.rstrip('\n')
                                            name, value = line1.split("=")
                                            exec(str(line), globals())
                                    copyfile(filename, tmpfile)
                                    from fssh_6 import runSim, genviz
                                    if not (os.path.exists(calcdir)):
                                        os.mkdir(calcdir)
                                    if not (viz_only):
                                        runSim()
                                    genviz()
                                    os.remove(tmpfile)
                                    del runSim
                                    del genviz
                                    del sys.modules['fssh_6']
        else:
            print('Deprecated')
            os.remove(tmpfile)
            sys.exit()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
