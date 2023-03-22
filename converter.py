import sys
from preprocess import load_song_in_kern


def main(in_krnfile='deutschl/erk/deut0567.krn', out_midifile ='midi/deut0567.mid'):
    score = load_song_in_kern(in_krnfile)
    score.write('midi', fp=out_midifile)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])