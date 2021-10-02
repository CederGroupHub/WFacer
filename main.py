"""Main program for CEAuto."""

__author__ = "Fengyu Xie"

from .CEAuto import InputsWrapper, HistoryWrapper
from .CEAuto import TimeKeeper
from .CEAuto import DataManager
from .CEAuto import StructureEnumerator
from .CEAuto import Featurizer
from .CEAuto import CEFitter
from .CEAuto import GSChecker

import logging

def main():
    """Main program.

    TimeKeeper enables hot restart.
    """
    while True:

        # Load and flush once for easier realoading later.
        iw = InputsWrapper.auto_load()
        tk = TimeKeeper.auto_load()
        if tk.iter_id < 1:
            hw = HistoryWrapper(iw.subspace)
        else:
            hw = HistoryWrapper.auto_load()

        # Flush parsed options for easy reading next time.
        iw.auto_save()
        hw.auto_save()
        tk.auto_save()

        # Passed down, and changed on-the-fly.
        dm = DataManager.auto_load()

        # Enumeration.
        if tk.todo('enum'):
            enum = StructureEnumerator(dm, hw)
            _ = enum.generate_structures(iter_id=tk.iter_id)
            dm = enum.data_manager.copy()
            dm.auto_save()
            tk.advance()
            tk.auto_save()

        # Write DFT inputs.
        if tk.todo('write'):
            writer = iw.calc_writer
            writer.write_df_entree(dm)
            dm.auto_save()
            tk.advance()
            tk.auto_save()

        # Submit and manage DFT calcs.
        if tk.todo('calc'):
            manager = iw.calc_manager
            manager.run_df_entree(dm)
            dm.auto_save()
            tk.advance()
            tk.auto_save()

        # Featurize.
        if tk.todo('feat'):
            feat = Featurizer(dm, hw)
            feat.featurize()
            feat.get_properties()
            dm = feat.data_manager.copy()
            dm.auto_save()
            tk.advance()
            tk.auto_save()

        # Fit energy CE.
        if tk.todo('fit'):
            fitter = CEFitter(dm, hw)
            fitter.fit()
            hw = fitter.history_wrapper.copy()
            hw.auto_save()
            tk.advance()
            tk.auto_save()

        # Check CE convergence
        if tk.todo('check'):
            checker = GSChecker(dm, hw)
            if checker.check_convergence():
                break

    logging.log("Congradulations! Your CE has reached convergence!")


if __name__=="__main__":
    CEAuto_run()
