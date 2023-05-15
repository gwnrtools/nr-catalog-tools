""" Helper and diagnostic function for tests """

import numpy as np
import config
import datetime
from inspect import getframeinfo, getmodule, stack
import os


def message(
    *args, message_verbosity=2, print_verbosity=config.print_verbosity, log_verbosity=config.log_verbosity, **kwargs
):
    """The print function with verbosity levels and logging facility.

    Notes
    -----

    Verbosity choices:

                    message_verbosity   :   Each message carries with it a verbosity level. More the verbosity more the priority. Default value is 2

                    print_verbosity     :   prints all messages above this level of verbosity.

                    log_verbosity       :   logs all messages above this level of verbosity.

    Verbosity  levels:

                    0: Errors

                    1: Warnings

                    2: Information

    Parameters
    ----------

                    ``*args`            :   non-keyword arguments
                                                                                                                    same arguments as to that of the print functions,

                    message_verbosity   :   int

                    print_verbosity     :   int

                    log_verbosity       :   int

                    ``**kwargs``        :   keyword arguments
                                                                                                                    Same as that of the print function.

    Returns
    -------

                    1                   :   int
                                                                                                                    messages to stdout and logging of messages, while the function returns 1."""

    # If message verbosity matches the global verbosity level, then print
    if message_verbosity <= print_verbosity:
        print(*args, **kwargs)
    if log_verbosity <= message_verbosity:
        now = str(datetime.datetime.now())
        tstamp = (now[:10] + "_" + now[11:16]).replace(':', '-')
        caller = getframeinfo(stack()[1][0])
        # frameinfo = getframeinfo(currentframe())
        if not os.path.isdir("logs"):
            os.mkdir("logs")

        with open("logs/" + tstamp + ".log", "a") as log_file:
            if message_verbosity == -1:
                for line in traceback.format_stack():
                    log_file.write(line.strip())
            log_file.write("\n")
            log_file.write("{}:{}\t{}".format(caller.filename, caller.lineno, *args))
            log_file.write("\n")
    return 1


def rms_errs(func1, func2, Norm=False):
    """Compute and return the error estimates between two arrays

    Parameters
    ----------
    func1, func2 : ndarray
                   Arrays of same shape to compare with.
    info : sphericalarray
           Grid info

    Returns
    -------
    RMS : float
          The RMS error
    Amax : float
           The max diff relative to A1max
    Amin : float the min diff relative to A2max
    """
    # Normalize the waveforms wrt the first
    n1 = np.linalg.norm(func1)
    func1 = func1/n1
    func2 = func2/n1

    A1max = np.amax(np.absolute(func1))

    diff = func1 - func2

    Amax = np.amax(diff) / A1max
    Amin = np.amin(diff) / A1max

    RMS = np.sqrt(np.sum(np.absolute(diff) ** 2) / len(func1)) / A1max

    return RMS, Amin, Amax
