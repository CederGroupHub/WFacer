"""
Main program for CEAuto.
"""

__author__ = "Fengyu Xie"

#Status checker not explicitly included! Every sub-module
#will automatically check status and skip if already finished
#In current iteration!
from CEAuto import InputsWrapper
from CEAuto import DataManager

from CEAuto import StructureEnumerator
from CEAuto import Featurizer
from CEAuto import Fitter
from CEAuto import GSChecker
from CEAuto import GSGenerator

def main():
    #Load and flush once for easier realoading later.
    iwrapper = InputsWrapper.auto_load()
    iwrapper.auto_save()
    while True:

        enum = StructureEnumerator.auto_load()
        _ = enum.generate_structures()
        enum.auto_save()

        writer = InputsWrapper.auto_load().get_calc_writer(DataManager.auto_load())
        writer.auto_write_entree()

        manager = InputsWrapper.auto_load().get_calc_manager(DataManager.auto_load())
        manager.auto_run()

        feat = Featurizer.auto_load()
        feat.featurize()
        feat.auto_save()

        fitter = Fitter.auto_load()
        fitter.fit()
        fitter.auto_save()

        #An iteration actually starts here. But for the first
        #iteration, it must start at enumerator.
        checker = GSChecker.auto_load()
        if checker.check_convergence():
            break

        gen = GSGenerator.auto_load()
        gen.solve_gss()
        gen.auto_save()

    print("Congradulations! CE has reached convergence!")

if __name__=="__main__":
    main()
