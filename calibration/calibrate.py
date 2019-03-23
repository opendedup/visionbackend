#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import codecs, json

# local modules
from common import splitfn
from scipy.spatial import distance as dist

# built-in modules
import os

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)
    if not img_mask:
        img_mask = '../data/left??.jpg'  # default
    else:
        img_mask = img_mask[0]

    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))

    pattern_size = (7, 4)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    cam = cv.VideoCapture('/dev/video2')
    cam.set(cv.CAP_PROP_FRAME_WIDTH,2560);
    cam.set(cv.CAP_PROP_FRAME_HEIGHT,1440);
    while True:
        ret_val, img = cam.read()

        cv.imshow('my webcam', img)
        if cv.waitKey(1) == 27:
            break  # esc to quit
    cv.destroyAllWindows()
    ret, img = cam.read()
    h, w = img.shape[:2]  # TODO: use imquery call to retrieve results
    print(img.shape[:2])
    def processImage(camera,x):
        while True:
            ret_val, img = camera.read()

            cv.imshow('my webcam', img)
            if cv.waitKey(1) == 27:
                break  # esc to quit
        print('processing %s... ' % camera)
        ret, img = camera.read()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img is None:
            print("Failed to load", camera)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            print(corners[0][0])
            print(corners[0][0][0])
            print(corners[0][0][1])

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            outfile = os.path.join(debug_dir, str(x) + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('chessboard found for %s ' % camera )
        return (corners.reshape(-1, 2), pattern_points)

    threads_num = int(args.get('--threads'))
    chessboards = [processImage(cam,x) for x in range(4)]

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
    print(obj_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    config = {'rms':rms,'camera_matrix':camera_matrix.tolist(),'dist_coefs_shape': dist_coefs.shape,'dist_coefs':dist_coefs.ravel().tolist()}

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix.tolist())
    print("distortion coefficients: ", dist_coefs.ravel())
    print(json.dumps(config))
    # undistort the image with the calibration
    print('')
    for x in range(4) if debug_dir else []:
        img_found = os.path.join(debug_dir, str(x) + '_chess.png')
        outfile = os.path.join(debug_dir, str(x) + '_undistorted.png')

        ret, img = cam.read()
        if img is None:
            continue

        h, w = img.shape[:2]
        print('h={},w={}'.format(h,w))
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, pattern_size)

        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

            ca = corners[0]
            for x in range(len(corners)):
                if (x+2) > len(corners):
                    break

                nxt = x+1
                D = dist.euclidean((corners[x][0][0], corners[x][0][1]), (corners[nxt][0][0], corners[nxt][0][1]))
                print(D)

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    cv.destroyAllWindows()
