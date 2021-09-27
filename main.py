"""
Main program for CEAuto.
"""

__author__ = "Fengyu Xie"

# Status checker not explicitly included! Every sub-module
# will automatically check status and skip if already finished
# In current iteration!
from CEAuto import InputsWrapper
from CEAuto import DataManager
from CEAuto import StructureEnumerator
from CEAuto import Featurizer
from CEAuto import CEFitter
from CEAuto import GSChecker
from CEAuto import GSGenerator

import logging

def CEAuto_run():
    # Load and flush once for easier realoading later.
    iwrapper = InputsWrapper.auto_load()

    # Flush parsed options for easy reading next time.
    iwrapper.auto_save()

    while True:
        # Passed down, and changed on-the-fly.
        dm = DataManager.auto_load()

        enum = StructureEnumerator.auto_load(dm)
        _ = enum.generate_structures()
        enum.auto_save()

        writer = InputsWrapper.auto_load().calc_writer
        writer.write_df_entree(dm)
        dm.auto_save()

        manager = InputsWrapper.auto_load().calc_manager
        manager.run_df_entree(dm)
        dm.auto_save()

        feat = Featurizer.auto_load(dm)
        feat.featurize()
        feat.get_properties()
        feat.auto_save()

        fitter = CEFitter.auto_load(dm)
        fitter.fit()
        fitter.auto_save()

        # An iteration actually starts here. But for the first
        # iteration, it must start at enumerator.
        checker = GSChecker.auto_load(dm)
        if checker.check_convergence():
            break

        gen = GSGenerator.auto_load(dm)
        gen.solve_gss()
        gen.auto_save()

    logging.log("Congradulations! CE has reached convergence!")

if __name__=="__main__":
    CEAuto_run()
