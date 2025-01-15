from finish import Finisher
import argparse

def run(path=None, clean=False, flush=False, anonymize=False):
    r = Finisher(path, clean, flush, anonymize)
    r.run()

if __name__ =="__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Choose finish and clean setup")
    parser.add_argument("--clean", required=False, help="Eliminate descriptors")
    parser.add_argument("--flush", required=False, help="Eliminate the estimators")
    parser.add_argument("--anonymize", required=False, help="Eliminate all SMILES")
    args = parser.parse_args()
    clean = args.clean
    flush = args.flush
    anonymize = args.anonymize
    run(path=None, clean=clean, flush=flush, anonymize=anonymize)