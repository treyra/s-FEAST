"""
Code for taking pdf and making a gif. Credit Ben Riviere

Should be run as a toplevel method
"""
import os
import subprocess
import glob
import multiprocessing as mp

def makeImage(iPage):
    """
    Converts the specified page of the pdf to a png image.
    """

    makeImageCommand = f"convert -density 400 {INPUT_PDF_PATH}[{iPage}] {makeTempImageName(iPage)}" #Using globals in utility function pylint: disable=used-before-assignment
    runCommand(makeImageCommand)


def runCommand(command):
    """
    Run a string as a terminal command

    Parameters
    ----------
    command : string
        Command to run
    """
    print(f'running cmd: {command}...')
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) as process:
        process.wait()
    print(f'completed cmd: {command}')


def makeTempImageName(iPage):
    """
    Creates a temporary name for each image, which is a 4 digit number prepended by "x-"
    For example, if iPage = 2, returns "TEMP_IMAGES_DIR/x-0002.png"
    """
    return f"{TEMP_IMAGES_DIR}/x-{iPage:04d}.png" #Using globals in utility function pylint: disable=used-before-assignment


def makeMovieFromPdf(pdfPath, videoPath):
    """
    Converts the specified pdf into a gif
    """

    #Check for and remove any previously existing temp images
    if os.path.exists(TEMP_IMAGES_DIR):
        clearTempDirectory(TEMP_IMAGES_DIR)
    else:
        os.mkdir(TEMP_IMAGES_DIR)


    # # make images (serial)
    # for i in range(getNumPages(pdf_path)):
        # makeImageCommand = "convert -density 400 {}[{}] {}".format(pdf_path, i, tii(i))
        # runCommand(makeImageCommand)


    # make images (parallel)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.map(makeImage, range(getNumPages(pdfPath)))


    # make movie
    frameRate = 30  # frames per second
    #Need to  have ffmpeg installed, creates video using all the matching temp images we made that match the regex.
    makeMovieCommand = f"ffmpeg -r {frameRate} -i {TEMP_IMAGES} -c:v libx264 -r 30 -y -pix_fmt yuv420p {videoPath}" #Using globals in utility function pylint: disable=used-before-assignment
    runCommand(makeMovieCommand)

    #Clean up
    clearTempDirectory(TEMP_IMAGES_DIR)

def getNumPages(pdfPath):
    """
    Uses pdfinfo to parse the pdf to find the number of pages
    Need to have pdfinfo installed, should be by default on Ubuntu
    """
    output = subprocess.check_output(["pdfinfo", pdfPath]).decode()
    pagesLine = [line for line in output.splitlines() if "Pages:" in line][0]
    numPages = int(pagesLine.split(":")[1])
    return numPages

def clearTempDirectory(tempDirectory):
    """
    Deletes every file in the specified temporary directory
    """
    files = glob.glob(tempDirectory + "/*")
    for fileName in files:
        os.remove(fileName)

#If running directly
if __name__ == "__main__":
    INPUT_PDF_PATH = None #SET TO PDF TO TURN INTO MOVIE
    OUTPUT_VIDEO_PATH = None #SET TO OUTPUT PATH FOR MP4

    # prep temporary directory
    TEMP_IMAGES_DIR = f"{os.path.dirname(os.path.realpath(INPUT_PDF_PATH))}/Temp"
    #The generated temp images will be of form "TEMP_IMAGES_DIR/x-%04d.png"
    #where %04d is a 4 digit number. ie, for image 2, it would be: "TEMP_IMAGES_DIR/x-0002.png"
    TEMP_IMAGES = f"{TEMP_IMAGES_DIR}/x-%04d.png"


    makeMovieFromPdf(INPUT_PDF_PATH, OUTPUT_VIDEO_PATH)
else:
    #This module shouldn't be imported
    raise ImportError("This module is intended as a stand alone utility and is not configured for importing.")
