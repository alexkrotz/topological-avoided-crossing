import sys
from shutil import copyfile

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

    if dim == 2:
        # run 2-D code
        if sim == 'WP':
            # run wavepacket dynamics
            from wp import runSim
            psi_out = runSim()
            #runsim()
            sys.exit()
        if sim == 'MF':
            # run mean-field dynamics
            print('Not implemented')
            sys.exit()
        if sim == 'FSSH':
            # run FSSH dynamics
            print('Not implemented')
            sys.exit()
    if dim == 1:
        # run 1-D code
        if sim == 'WP':
            # run wavepacket dynamics
            print('Not implemented')
            sys.exit()
        if sim == 'MF':
            # run mean-field dynamics
            print('Not implemented')
            sys.exit()
        if sim == 'FSSH':
            # run FSSH dynamics
            print('Not implemented')
            sys.exit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
