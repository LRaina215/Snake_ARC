"""Rebuild the single-file real entry from shared production sources; preserve user parameters."""
from build_real_feeding_core import main as feeding
from build_real_arc_helix import main as transitions
from build_real_curve_feeding import main as curves


def main():
    feeding()
    transitions()
    curves()


if __name__=='__main__':main()
