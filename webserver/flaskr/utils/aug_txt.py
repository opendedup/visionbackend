#=======================================================================#
# extract_data.py                                                       #
#=======================================================================#
# usage: extract_data.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR]
#
# This program extracts provision numbers from a set of documents.
#
# optional arguments:
#  -h, --help            show this help message and exit
#  -i INPUT_DIR, --input_dir INPUT_DIR
#                        Input directory for the files to be modified
#  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
#                        Output directory for the files to be modified
#=======================================================================#

#=======================================================================#
# Sample usage:                                                         #
#=======================================================================#
#   python extract_data.py --input_dir ocr/data/ --output_dir ocr/results/
#=======================================================================#


import numpy as np
import os
import cv2
import glob
import shutil
import pytesseract
import re
import time
import argparse
from statistics import mode
import io

regex = r"P\d{17}"
found = {}
results = {}
queue = []
done = []
missing = []
pnr_area = [150, 450, 1600, 1150]  # [start_x, start_y, end_x, end_y]


# =============================================================================== #
#    To-do list                                                                   #
# =============================================================================== #
# 0. Provision Number                                                             #
# =============================================================================== #


# =============================================================================== #
#    Threshold Methods                                                            #
# =============================================================================== #
# 1. Binary-Otsu w/ Gaussian Blur (kernel size = 9)                               #
# 2. Binary-Otsu w/ Gaussian Blur (kernel size = 7)                               #
# 3. Binary-Otsu w/ Gaussian Blur (kernel size = 5)                               #
# 4. Binary-Otsu w/ Median Blur (kernel size = 5)                                 #
# 5. Binary-Otsu w/ Median Blur (kernel size = 3)                                 #
# 6. Adaptive Gaussian Threshold (31,2) w/ Gaussian Blur (kernel size = 5)        #
# 7. Adaptive Gaussian Threshold (31,2) w/ Median Blur (kernel size = 5)          #
# =============================================================================== #

def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")


def crop_image(img, start_x, start_y, end_x, end_y):
    cropped = img[start_y:end_y, start_x:end_x]
    return cropped

def detect_text(img_path):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image,image_context={"language_hints": ["en"]})
    print('Texts:{}'.format(response.full_text_annotation.text))
    return response.full_text_annotation.text

def get_string(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cheight, cwidth = img.shape[:2]
    dim = [cwidth,cheight]
    maxdim =500
    if max(dim) < maxdim:
                            scalef = maxdim/max(dim)
                            swidth = int(img.shape[1] * scalef)
                            sheight = int(img.shape[0] * scalef)
                            sdim = (swidth, sheight)
                            zimg = cv2.resize(img, sdim, interpolation = cv2.INTER_CUBIC)
                            
    
    
    
    b_save_path = os.path.join(output_path, file_name + "_nofilter_" +str(maxdim) + "_"+ str(method) + ".png")
    cv2.imwrite(b_save_path, zimg)

    # Convert to gray
    zimg = cv2.cvtColor(zimg, cv2.COLOR_BGR2GRAY)
    bw_save_path = os.path.join(output_path, file_name + "_bwnofilter_" +str(maxdim) + "_" + str(method) + ".png")
    cv2.imwrite(bw_save_path, zimg)
    detect_text(bw_save_path)
    
    detect_text(b_save_path)
    
    detect_text(img_path)
    print("maxdim = "+str(maxdim))
    return ""


def find_match(regex, text):
    matches = re.finditer(regex, text, re.MULTILINE)
    target = ""
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1

        print("  Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                            end=match.end(), match=match.group()))
        target = match.group()

    return target


def pretty_print(result_dict):
    s = ''
    for key in result_dict:
        s += '# ' + key + ': ' + result_dict[key] + '\n'
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program extracts provision numbers from a set of documents.")
    parser.add_argument("-i", "--input_dir", help="Input directory for the files to be modified")
    parser.add_argument("-o", "--output_dir", help="Output directory for the files to be modified")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    im_names = glob.glob(os.path.join(input_dir, '*.png')) + \
               glob.glob(os.path.join(input_dir, '*.jpg')) + \
               glob.glob(os.path.join(input_dir, '*.jpeg'))

    overall_start_t = time.time()
    for im_name in sorted(im_names):
        queue.append(im_name)

    print("The following files will be processed and their provision numbers will be extracted: {}\n".format(queue))

    for im_name in im_names:
        start_time = time.time()
       
        queue.remove(im_name)
        file_name = im_name.split(".")[0].split("/")[-1]

        i = 1
        while i < 2:
            result = get_string(im_name, i)
            match = find_match(regex, result)
            if match:
                if file_name in found:
                    found[file_name].append(match)
                else:
                    list = []
                    list.append(match)
                    found[file_name] = list

            f = open(os.path.join(output_dir, file_name, file_name + "_filter_" + str(i) + ".txt"), 'w')
            f.write(result)
            f.close()
            i += 1

        pnr = ''
        if file_name in found:
            pnr = mode(found[file_name])
            results[file_name] = pnr
            done.append(file_name)
        else:
            missing.append(file_name)
        end_time = time.time()

        

    overall_end_t = time.time()

    print('#=======================================================\n'
          '# Summary \n'
          '#=======================================================\n'
          '# The documents that are successfully processed are: \n' + pretty_print(results) +
          '#=======================================================\n'
          '# The program failed to extract information from: \n' 
          '# ' + str(missing) + '\n'
          '#=======================================================\n'
          '# It took ' + str(overall_end_t-overall_start_t) + ' seconds.\n'
          '#=======================================================\n')