import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rosbag
import cv2
import pickle
import math
from cv_bridge import CvBridge
import argparse
from os.path import exists, dirname, splitext, basename

# Main function that should be executed ONLY when this script is executed from the command line
# If this script is executed from another Python script, use getSynchronizedImages below instead with appropriate args
def main():
    getCmdLineArgs()

def getCmdLineArgs():
    parser = argparse.ArgumentParser(description="Get Synchronized Images")
    parser.add_argument("--bagFilePath", type=str, help="""
        Path to the bagfile to extract images from. Ex. '/xxx/yyy/bagfile.bag'.
    """)
    parser.add_argument("--leftCameraTopic", type=str, default="/camera/left/image_raw", help="""
        The name of the topic containing the left camera images. Ex. '/left_camera/image'. If not provided, default is 
        '/camera/left/image_raw'.
    """)
    parser.add_argument("--rightCameraTopic", type=str, default="/camera/right/image_raw", help="""
        The name of the topic containing the right camera images. Ex. '/right_camera/image'. If not provided, default is 
        '/camera/right/image_raw'.
    """)
    parser.add_argument("--leftPickleFilePath", type=str, help="""
        Path to pickle file containng left camera infomration from a bagfile. Ex. '/xxx/yyy/left.pickle'. 
    """)
    parser.add_argument("--rightPickleFilePath", type=str, help="""
        Path to pickle file containng right camera infomration from a bagfile. Ex. '/xxx/yyy/right.pickle'. 
    """)
    parser.add_argument("--saveDataToPickle", action="store_true", help="""
        Whether to save the timestamp data extracted from the bagfile in a pickle so it can be loaded in quicker the next time.
        From the command line, specify "yes" or "no" (case-insensitive). From another program, specify True or False. Default value 
        is "no"/False. If this option is specified "yes"/True, then the --bagFileOption, --leftPickleFilePath, and 
        --rightPickleFilePath options MUST also be specified.
    """)
    parser.add_argument("--getDataFromPickle", action="store_true", help="""
        Whether to get the timestamp data from a pickle file instead of the bagfile. From the command line, specify "yes" or "no" 
        (case-insensitive). From another program, specify True or False. Default value is "no"/False. If this option is specified 
        "yes"/True, then the --bagFileOption can be omitted given the --matchedImagesFolderPath option is also NOT specified.
    """)
    parser.add_argument("--threshold", type=float, default=0.05, help="""
        Threshold in seconds for matches. Only matches with a time difference less than this threshold will be output. Ex. '0.01'. 
        Default threshold is 0.05.
    """)
    parser.add_argument("--matchesCSVFilePath", type=str, help="""
        Path to the CSV file where the matched timestamps and indexes should be stored. Ex. '/xxx/yyy/matches.csv'. If not specified,
        matches will not be exported.
    """)
    parser.add_argument("--matchedImagesFolderPath", type=str, help="""
        Path to the folder where the matched images will be saved. Ex. '/xxx/yyy/zzz'. If not specified, matched images will not be 
        saved. If this option is specified, --bagFilePath must also be specified to be able to get the images from the bagfile. Note:
        A trailing slash should NOT be provided to this argument.
    """)
    parser.add_argument("--matchedImagesFileNamePrefix", type=str, help="""
        The prefix of the file names of the matched images that are saved to a directory. Ex. If 'xxxyyy' is provided for this argument, 
        the image file names would look like "xxxyyy_left_frame58_pair34.jpg" and "xxxyyy_right_frame65_pair34.jpg". If this option is not 
        specified, the name of the bagfile will be used as a default. Note: This option is only used if --matchedImagesFolderPath 
        is set.
    """)
    parser.add_argument("--includeTimeDiffInMatchedImagesFileName", action="store_true", help="""
        Whether to include the time difference of the match pair an image belongs to in its filename. rom the command line, specify 
        "yes" or "no" (case-insensitive). From another program, specify True or False. Default value is "no"/False. If this option
        is specified "yes"/True then the image file names would look like "xxxyyy_left_frame58_pair34_diff19486039.jpg" where 19486039 
        is the value leftTime-rightTime (in nanoseconds) associated with pair 34 in the matches. If this value is negative, the image
        file name may look like "xxxyyy_left_frame58_pair34_diff-19486039.jpg". If this option is specified "no"/False, the part of 
        the image file name including and following "_diff" will not be included.

    """)
    parser.add_argument("--matchDiffsHistogramFilePath", type=str, help="""
        Path to the PNG file where a hisogram of the time differences in the matches will be stored. Ex. '/xxx/yyy/histogram.png'. If 
        this argument is provided, the histogram will be generated and saved after all other actions are complete (i.e. after the CSV 
        file and matched images are saved if those options were specified).
    """)
    args = parser.parse_args()
    getSynchronizedImages(bagFilePath=args.bagFilePath, leftCameraTopic=args.leftCameraTopic, rightCameraTopic=args.rightCameraTopic, leftPickleFilePath=args.leftPickleFilePath, rightPickleFilePath=args.rightPickleFilePath, saveDataToPickle=args.saveDataToPickle, getDataFromPickle=args.getDataFromPickle, threshold=args.threshold, matchesCSVFilePath=args.matchesCSVFilePath, matchedImagesFolderPath=args.matchedImagesFolderPath, matchedImagesFileNamePrefix=args.matchedImagesFileNamePrefix, includeTimeDiffInMatchedImagesFileName=args.includeTimeDiffInMatchedImagesFileName, matchDiffsHistogramFilePath=args.matchDiffsHistogramFilePath)

# This is the primary function that drives this entire program and returns a tuple like so: (# left frames, # right frames, # matches)
def getSynchronizedImages(bagFilePath=None, leftCameraTopic=None, rightCameraTopic=None, leftPickleFilePath=None, rightPickleFilePath=None, saveDataToPickle=None, getDataFromPickle=None, threshold=None, matchesCSVFilePath=None, matchedImagesFolderPath=None, matchedImagesFileNamePrefix=None, includeTimeDiffInMatchedImagesFileName=None, matchDiffsHistogramFilePath=None):
    
    # Set left and right camera topics and threshold to default value if not specified
    if not leftCameraTopic:
        leftCameraTopic = "/camera/left/image_raw"
    if not rightCameraTopic:
        rightCameraTopic = "/camera/right/image_raw"
    if not threshold:
        threshold = 0.05

    # Make sure the user specified a bagFilePath if they also specified a matchedImagesFolderPath
    assert(not(matchedImagesFolderPath and not bagFilePath)), "You must specify a bagFilePath if you specify a matchedImagesFolderPath."

    # Make sure the folder where the left pickle file will be saved exists
    if leftPickleFilePath:
        assert(exists(dirname(leftPickleFilePath))), "The directory where the left pickle file was to be saved does not exist."

    # Make sure the folder where the right pickle file will be saved exists
    if rightPickleFilePath:
        assert(exists(dirname(rightPickleFilePath))), "The directory where the right pickle file was to be saved does not exist."

    # Make sure the folder where the matched images will be saved exists
    if matchedImagesFolderPath:
        assert(exists(matchedImagesFolderPath)), "The directory where the matched images were to be saved does not exist."

    # Make sure the folder where the CSV file will be saved exists
    if matchesCSVFilePath:
        assert(exists(dirname(matchesCSVFilePath))), "The directory where the CSV file was to be saved does not exist."
    
    # Make sure the folder where the histogram image will be saved exists
    if matchDiffsHistogramFilePath:
        assert(exists(dirname(matchDiffsHistogramFilePath))), "The directory where the histogram image was to be saved does not exist."
    
    # Arrays to hold the left and right timestamps
    left = []
    right = []
    
    # If the left and right pickleFilePaths were specified and exist and the user wants to get the data from the pickle files, then do so
    if leftPickleFilePath and rightPickleFilePath and getDataFromPickle:
        
        # Make sure pickle files exist at the provided paths
        assert(exists(leftPickleFilePath)), "The left pickle file does not exist at the given path."        
        assert(exists(rightPickleFilePath)), "The right pickle file does not exist at the given path."

        # Get the data from the pickle files
        left, right = getTimestampsFromPickle(leftPickleFilePath, rightPickleFilePath)

    # If the user does not want to get data from pickle files and the bagFilePath was specified and exists, get the data from the bagfile
    elif bagFilePath:

        # Make sure bagfile exists at the provided path
        assert(exists(bagFilePath)), "The bagfile does not exist at the given path."

        # Get the data from the bagfile
        left, right = getDataFromBagfile(bagFilePath, leftCameraTopic, rightCameraTopic)

        # If the left and right pickleFilePaths were specified and user wants to save the data to pickle files, then do so at the specified locations
        if leftPickleFilePath and rightPickleFilePath and saveDataToPickle:
            saveTimestampsToPickle(leftPickleFilePath, rightPickleFilePath, left, right)

    else:
        raise Exception("Data could not be retrieved because an invalid set of arguments have been passed to this script. See the help menu (-h) for more details on correct combinations of arguments.")

    # Print the lengths of the arrays, which give the number of frames in each array
    print("Number of left frames: " + str(len(left)))
    print("Number of right frames: " + str(len(right)))

    # Get the matched timestamps
    matches = getMatches(left, right, threshold)

    # Print the number of matches
    print("Number of matches that met the threshold: " + str(len(matches.index)))

    # If the user wants to save the matches to a CSV, do so
    if matchesCSVFilePath:
        saveMatchesToCSV(matches, matchesCSVFilePath)

    # If the user wants to save the matched images in a folder, do so
    if matchedImagesFolderPath:
        if not matchedImagesFileNamePrefix:
            matchedImagesFileNamePrefix = getBagFileName(bagFilePath)

        saveMatchedImages(bagFilePath, leftCameraTopic, rightCameraTopic, matches, matchedImagesFolderPath, matchedImagesFileNamePrefix, includeTimeDiffInMatchedImagesFileName)
    
    # If the user wants to save a histogram of the matched time differences, do so
    if matchDiffsHistogramFilePath:
        showMatchesHistogram(matches, matchDiffsHistogramFilePath)

    # Return a tuple with the information that was printed out above beacause if this function is called from another program (not command line),
    # it does not get printed information, so returning this tuple ensures it still gets this information
    return (len(left), len(right), len(matches.index))

# Function to get the left and right timestamps from a bagfile
def getDataFromBagfile(bagFilePath, leftCameraTopic, rightCameraTopic):

    # Interpret the bagfile as a Bag object
    bag = rosbag.Bag(bagFilePath)

    # Arrays to hold the left and right timestamps that will be returned
    left = []
    right = []

    # Go through all of the messages in the bagfile and append the time values and their frame indexes to the arrays
    for topic, msg, t in bag.read_messages(topics=[leftCameraTopic, rightCameraTopic]):
        if(topic == leftCameraTopic):
            left.append(t.to_nsec())
        elif(topic == rightCameraTopic):
            right.append(t.to_nsec())
        else:
            print("A message from a topic other than " + leftCameraTopic + " or " + rightCameraTopic + " was found in the bagfile.")

    # Close the bagfile
    bag.close()

    return (left, right)

# Function to save the left and right timestamps to a pickle file
def saveTimestampsToPickle(leftPickleFilePath, rightPickleFilePath, left, right):

    # Save left data to pickle file at specified path
    with open(leftPickleFilePath, 'wb') as outfile:
        pickle.dump(left, outfile)

     # Save right data to pickle file at specified path
    with open(rightPickleFilePath, 'wb') as outfile:
        pickle.dump(right, outfile)

# Function to get the left and right timestamps from a pickle file
def getTimestampsFromPickle(leftPickleFilePath, rightPickleFilePath):
    
    # Arrays to hold the left and right times from the files 
    left = []
    right = []

    # Get the times saved in pickle files
    with open(leftPickleFilePath, 'rb') as infile:
        left = pickle.load(infile)

    with open(rightPickleFilePath, 'rb') as infile:
        right = pickle.load(infile)
    
    return (left, right)

# Function to get the matches into a dataframe from the left and right timestamps that are under the specified threshold
def getMatches(left, right, threshold):
    
    # Array to hold the matched time values
    matches = pd.DataFrame({"leftIndex": [], "leftTime": [], "rightIndex": [], "rightTime": [], "leftTime-rightTime": []})

    # Determine whether the left or right list has less frames, and use that as the base for finding the nearest values in the other list
    # Add the nearest values whose difference is under the threshold to the matches dataframe
    if(len(left) < len(right)):
        for leftTime in left:
            rightTime = findNearest(right, leftTime)
            diff = leftTime - rightTime
            if(abs(diff) < threshold * 1e9):
                matches.at[len(matches.index)] = [left.index(leftTime), leftTime, right.index(rightTime), rightTime, diff]
    else:
        for rightTime in right:
            leftTime = findNearest(left, rightTime)
            diff = leftTime - rightTime
            if(abs(diff) < threshold * 1e9):
                matches.at[len(matches.index)] = [left.index(leftTime), leftTime, right.index(rightTime), rightTime, diff]

    return matches

# Function to find the element in the array whose value is the closest to the provided value
def findNearest(array, value):

    index = np.searchsorted(array, value, side="left")
    if index > 0 and (index == len(array) or math.fabs(value - array[index-1]) < math.fabs(value - array[index])):
        return array[index - 1]
    else:
        return array[index]

# Save the matches dataframe as a CSV at the provided path
def saveMatchesToCSV(matches, matchesCSVFilePath):
    matches.to_csv(matchesCSVFilePath, float_format="%.0f")

# Get the name of the bagfile without extension from a path to a bagfile
def getBagFileName(bagFilePath):
    return splitext(basename(bagFilePath))[0]

# Save the matched images into a directory
def saveMatchedImages(bagFilePath, leftCameraTopic, rightCameraTopic, matches, matchedImagesFolderPath, matchedImagesFileNamePrefix, includeTimeDiffInMatchedImagesFileName):

    # Interpret the bagfile as a Bag object
    bag = rosbag.Bag(bagFilePath)

    # Create a CvBridge object
    bridge = CvBridge()

    # Loop through all the images in the bagfile and save each image that is part of matches
    for topic, msg, t in bag.read_messages(topics=[leftCameraTopic, rightCameraTopic]):
        camera = "left" if topic == leftCameraTopic else "right"
        if(index := matches.index[matches[camera + "Time"] == t.to_nsec()].tolist()):
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            imgName = matchedImagesFolderPath + "/" + matchedImagesFileNamePrefix + "_" + camera + "_frame" + str(int(matches.at[index[0], camera + "Index"])) + "_pair" + str(index[0])
            if(includeTimeDiffInMatchedImagesFileName): 
                imgName = imgName + "_diff" + str(int(matches.at[index[0], "leftTime-rightTime"]))
            imgName = imgName + ".jpg"

            cv2.imwrite(imgName, cv_image)            

# Create and save a histogram of match time differences
def showMatchesHistogram(matches, matchDiffsHistogramFilePath):

    fig, ax = plt.subplots()
    ax.hist(matches["leftTime-rightTime"].div(1e9), bins=np.arange(-0.2, 0.2, 0.01), rwidth=0.8)
    plt.xlabel("leftTime - rightTime (sec)")
    plt.ylabel("Frequency (#)")
    fig.savefig(matchDiffsHistogramFilePath)

if __name__ == '__main__':
    main()