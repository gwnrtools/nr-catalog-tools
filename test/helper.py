''' Helper and diagnostic function for tests '''

import numpy as np
import config
import datetime
from inspect import getframeinfo, getmodule, stack
import os
def message(
    *args, message_verbosity=2, print_verbosity=config.print_verbosity, log_verbosity=config.log_verbosity, **kwargs):
    ''' The print function with verbosity levels and logging facility.
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
                    message_verbosity   :   int
                    print_verbosity     :   int
                    log_verbosity       :   int
                    ``**kwargs``        :   keyword arguments
                                            Same as that of the print function.
    Returns
    -------
                    1                   :   int                                                                     
                                            messages to stdout and logging of messages, while the function returns 1.'''

    # If message verbosity matches the global verbosity level, then print
    if message_verbosity <= print_verbosity:
        print(*args, **kwargs)
    if log_verbosity <= message_verbosity:
        now = str(datetime.datetime.now())
        tstamp = now[:10] + "_" + now[11:16]
        caller = getframeinfo(stack()[1][0])
        # frameinfo = getframeinfo(currentframe())
        if not os.path.isdir("logs"):
            os.mkdir("logs")
    
        #frame = stack()[1]
        #module = getmodule(frame[0])
        #fname = module.__file__
        fname = caller
        #fname = os.path.basename(__file__)
        with open(f"logs/" + tstamp + ".log", "a") as log_file:
            if message_verbosity == -1:
                log_file.write('\n')
                log_file.write(caller)
                for line in traceback.format_stack():
                    log_file.write(line.strip())
            log_file.write("\n")
            log_file.write("{}:{}\t{}".format(caller.filename, caller.lineno, *args))
            log_file.write("\n")
    return 1



def TransformSpinsNRtoLAL(nrSpin1, nrSpin2, n_hat, ln_hat):
    ''' Trnasform the spins of the NR simulation from the
    NR frame to the  frame.
    Parameters
    ---------
    nrSpin1, nrSpin2 : list
             A list of the components of the spins of the objects.
    nhat, ln_hat : list
             A list of the components of the unit vectors of the objects, 
             against which the components of the spins are specified.
    Returns
    -------
    S1, S2 : list
             The transformed spins in LAL frame.
    '''
    nrSpin1x, nrSpin1y, nrSpin1z = nrSpin1
    nrSpin2x, nrSpin2y, nrSpin2z = nrSpin2
    
    n_hat_x, n_hat_y, n_hat_z = n_hat
    ln_hat_x, ln_hat_y, ln_hat_z = ln_hat
        
    S1x = nrSpin1x * n_hat_x + nrSpin1y * n_hat_y + nrSpin1z * n_hat_z
    
    S1y = nrSpin1x * (-ln_hat_z * n_hat_y + ln_hat_y * n_hat_z)\
         + nrSpin1y * (ln_hat_z * n_hat_x - ln_hat_x * n_hat_z) \
         + nrSpin1z * (-ln_hat_y * n_hat_x + ln_hat_x * n_hat_y)
            
    S1z = nrSpin1x * ln_hat_x + nrSpin1y * ln_hat_y + nrSpin1z * ln_hat_z
  
    S2x = nrSpin2x * n_hat_x + nrSpin2y * n_hat_y + nrSpin2z * n_hat_z
    S2y = nrSpin2x * (-ln_hat_z * n_hat_y + ln_hat_y * n_hat_z) \
         + nrSpin2y * (ln_hat_z * n_hat_x - ln_hat_x * n_hat_z) \
         + nrSpin2z * (-ln_hat_y * n_hat_x + ln_hat_x * n_hat_y)
            
    S2z = nrSpin2x * ln_hat_x + nrSpin2y * ln_hat_y + nrSpin2z * ln_hat_z
    
    S1 = [S1x, S1y, S1z]
    S2 = [S2x, S2y, S2z]
    
    return S1, S2


def RMSerrs(func1, func2):
    ''' Compute and return the error estimates between two arrays
    
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
    '''
    A1max = np.amax(np.absolute(func1))

    diff = (func1 - func2)

    Amax = np.amax(diff)/A1max
    Amin = np.amin(diff)/A1max

    RMS = np.sqrt(np.sum(np.absolute(diff)**2)/len(func1))/A1max

    return RMS, Amin, Amax
