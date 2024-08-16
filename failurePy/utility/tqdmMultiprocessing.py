"""
File with functions for using tqdm with multiprocessing or recursion to give interpretable results.
"""

from queue import Empty #Can't get this directly from mp.Queue
from tqdm import tqdm

def initializeProcessTqdm(rank,total):
    """
    Initializes the progress bar for each process. Only does so if at the top level or in the first process.

    Parameters
    ----------
    rank : int
        Level or the process call, 0 if top level for first process
    total : int
        Total number of steps the bar should be tracking

    Returns
    -------
    progressBar : tqdmProgressBar
        Progress bar object to be updated
    """
    progressBar = None
    if rank == 0:
        progressBar = initializeTqdm(total=total)
    return progressBar

def initializeTqdm(total):
    """
    Initializes the progress bar.

    Parameters
    ----------
    total : int
        Total number of steps the bar should be tracking

    Returns
    -------
    progressBar : tqdmProgressBar
        Progress bar object to be updated
    """

    return tqdm(total=total)


def updateTqdmMultiProcess(processID,totalPerWorker,queueObject,progressBar):
    """
    Updates the progress bar, by either updating the display (top level or first process)
    or putting the an update into the queue for the first process to update
    Only needed by the multiprocessing methods

    Parameters
    ----------
    processID : int
        Number representing the process call, 0 if top level for first process
    total : int
        Total number of steps the bar should be tracking
    queue : multiprocessing queue
        Queue object to send progress updates through
    progressBar : tqdmProgressBar
        Progress bar object to be updated
    """
    if processID == 0:
        count = totalPerWorker
        try:
            while True:
                count += queueObject.get_nowait()
        except Empty:
            pass
        progressBar.update(count)
    else:
        queueObject.put_nowait(totalPerWorker)
