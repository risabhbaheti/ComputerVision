# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys
from skimage.color.colorconv import rgb2gray
from cv2 import NORM_TYPE_MASK
from skimage.color.rgb_colors import green
from numpy import dtype

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    
    #ref_white = cv2.resize(cv2.imread("images_new/aligned000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)
    #ref_black = cv2.resize(cv2.imread("images_new/aligned001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)
    
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region
    # print ref_white
    # print ref_black
    # print ref_avg
    h, w = ref_white.shape
    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))
        
    scan_bits = np.zeros((h, w), dtype=np.uint16)
    
    #base_image = cv2.imread("images_new/aligned001.jpg", cv2.IMREAD_COLOR)
    base_image = cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR)
    
    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        #patt_gray = cv2.resize(cv2.imread("images_new/aligned%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)
        # print patt_gray.shape
        
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        # print bit_code
        # break
        # TODO: populate scan_bits by putting the bit_code according to on_mask
        for i in range(0, h):
            for j in range(0, w):
                if(on_mask[i][j] == True):
                    scan_bits[i][j] = scan_bits[i][j] + bit_code
    """count =0
    for i in range(len(scan_bits)):
        for j in range(len(scan_bits[i])):
            if scan_bits[i][j]:
                count+=1
    """                        
    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)
    img = np.zeros((h, w, 3), np.float32)
    norm_img = np.zeros((h, w, 3), np.float32)
    
    camera_points = []
    rgb_vals = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code
            i, j = binary_codes_ids_codebook[scan_bits[y, x]]
            if i >= 1279 or j >= 799:  # filter
                continue
            camera_points.append(((x / 2.0), (y / 2.0)))
            blue = base_image[y, x, 0]
            green = base_image[y, x, 1]
            red = base_image[y, x, 2]
            rgb_vals.append((red, green, blue))
            projector_points.append((i, j))
            img[y, x] = [0, j, i]
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    img_a = img[:, :, 0]
    img_b = img[:, :, 1]
    img_c = img[:, :, 2] 
    # Extracting single channels from 3 channel image

    # normalizing per channel data:
    img_b = (img_b - np.min(img_b)) * 255 / (np.max(img_b) - np.min(img_b) )
    img_c = (img_c - np.min(img_c)) * 255 / (np.max(img_c) - np.min(img_c) )

    # putting the 3 channels back together:
    norm_img[:, :, 0] = img_a
    norm_img[:, :, 1] = img_b
    norm_img[:, :, 2] = img_c

    # cv2.normalize(img,norm_img,0,255, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    cv2.imwrite(sys.argv[1] + 'correspondence.jpg', norm_img)
    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    
    camera_point_reshaped = np.reshape(camera_points, (len(camera_points), 1, 2))
    rgb_val_reshaped = np.array(np.reshape(rgb_vals, (len(rgb_vals), 1, 3)), dtype=np.float32)
    projector_point_reshaped = np.array(np.reshape(projector_points, (len(projector_points), 1, 2)), dtype=np.float32)
    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']
    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
        projectorProjectionMatrix = np.concatenate((projector_R,projector_t), axis=1)
        cameraProjectionMatrix = np.array([ [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0] ], dtype=np.float32)
        undistCam = cv2.undistortPoints(camera_point_reshaped, camera_K, camera_d)
        undistProj = cv2.undistortPoints(projector_point_reshaped , projector_K, projector_d)
        points_3d_temp = cv2.triangulatePoints(cameraProjectionMatrix, projectorProjectionMatrix, undistCam, undistProj)
        points_3d = cv2.convertPointsFromHomogeneous(np.transpose(points_3d_temp))
        mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400)
        points = []
        rgb_points = []
        for i in range(len(mask)):
            if(mask[i] == True):
                points.append(points_3d[i])
                rgb_points_temp = np.concatenate((points_3d[i], rgb_val_reshaped[i]), axis=1)
                rgb_points.append(rgb_points_temp)
                
        points_final = np.array(points)
        rgb_points_final = np.array(rgb_points)
        

    # projectionMat = np.matmul(camera_K, projectionMatrix)
    # print projectionMat

    return points_final, rgb_points_final
	
	
def write_3d_points(points_3d):
    
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))
    
    return points_3d
 
def write_3d_points_rgb(rgb_points):
    
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(rgb_points.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name, "w") as f:
        for p in rgb_points:
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], p[0, 3], p[0, 4], p[0, 5])) 
    
    return rgb_points
                   
if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d, rgb_points = reconstruct_from_binary_patterns()
    write_3d_points_rgb(rgb_points)
    write_3d_points(points_3d)
