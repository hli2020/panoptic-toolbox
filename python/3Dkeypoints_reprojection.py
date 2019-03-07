import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'

# For camera projection (with distortion)
import panutils

# Edges between joints in the body skeleton
body_edges = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 9],
    [2, 6],
    [2, 12],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [12, 13],
    [13, 14],
])

# Setup paths
data_path = '../data/'
seq_name = '160422_ultimatum1'

hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
hd_img_path = data_path+seq_name+'/hdImgs/'

colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

# Load camera calibration parameters
with open(data_path+seq_name+'/calibration_{0}.json'.format(seq_name)) as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k, cam in cameras.items():
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3, 1))

box_inflate = 100
# Select HD frame
# hd_idx = 14963
# hd_idx = 9263
# hd_idx = 6733
hd_idx = 16683
hd_idx = 3508
hd_idx = 9453
# hd_idx = 11108   # IMPORTANT: confidence score < 0.1 and no show in (00_00) view
# hd_idx = 10873       # only ONE point (too small)
# hd_idx = 18493     # too small
# hd_idx = 21343    # too small
# hd_idx = 973    # ids are zero (it's ok)
hd_idx = 4208   # this image is all green (corrupted)
# hd_idx = 3837
# hd_idx = 3613
hd_idx = 3873
hd_idx = 281  # view 2, valid>0.1, super weired
hd_idx = 16100
hd_idx = 2063   # two "neck" joints overlap -> heat map
hd_idx = 18290   # overlap case
hd_idx = 11474
hd_idx = 13667
hd_idx = 18491
hd_idx = 10808
hd_idx = 23495

PLOT_ALL_JOINTS = False   # whether to show outside joints (valid or not)

cam_list = [
    (0, 0),
    # (0, 5),
    # (0, 10),
    # (0, 15),
    # (0, 20),
    # (0, 25),
    # (0, 30),
]
for entry in cam_list:
    # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera
    cam = cameras[entry]

    # Load the corresponding HD image
    image_path = hd_img_path+'{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'], cam['node'], hd_idx)
    im = plt.imread(image_path)

    # Reproject 3D Body Keypoint onto the HD/VGA camera
    plt.figure(figsize=(15, 15))
    plt.imshow(im)
    plt.title('3D Body Projection on HD view ({0}), seq {1}, im {2}'.format(
        cam['name'], seq_name, hd_idx))
    currentAxis = plt.gca()
    currentAxis.set_autoscale_on(False)

    try:
        # Load the json file with this frame's skeletons
        skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)

        x1, x2, y1, y2 = im.shape[1], 0, im.shape[0], 0
        # Cycle through all detected bodies
        for body in bframe['bodies']:

            if body['id'] in list(range(100)):
            # if body['id'] in [47, 46]:
                # There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
                # where c1 ... c19 are per-joint detection confidences
                skel = np.array(body['joints19']).reshape((-1, 4)).transpose()     # 4x19

                # Project skeleton into view (this is like cv2.projectPoints)
                # pt: 3x19
                pt = panutils.projectPoints(
                    skel[0:3, :], cam['K'], cam['R'], cam['t'], cam['distCoef'])

                # Show only dot points detected with confidence
                valid = skel[3, :] > 0.1

                # 1. DOT
                plt.plot(pt[0, valid], pt[1, valid], '.',
                         color=colors[body['id'] % 7], markersize=15,
                         markeredgecolor='w')

                # 2. Plot VALID edges for each bone
                for edge in body_edges:
                    if valid[edge[0]] and valid[edge[1]]:
                        plt.plot(pt[0, edge], pt[1, edge], color=colors[body['id'] % 7])

                # 3. Show *ALL/or valid* joint numbers
                for ip in range(pt.shape[1]):
                    curr_color = colors[body['id'] % 7]
                    if PLOT_ALL_JOINTS:
                        plt.text(pt[0, ip], pt[1, ip]-5, '{0}'.format(ip), color=curr_color)
                        if valid[ip] is False:
                            plt.plot(pt[0, ip], pt[1, ip], 'D',
                                     color=colors[body['id'] % 7], markersize=15, )
                    else:
                        if 0 <= pt[0, ip] < im.shape[1] and 0 <= pt[1, ip] < im.shape[0]:
                            plt.text(pt[0, ip], pt[1, ip] - 5, '{0}'.format(ip),
                                     color=curr_color)
                            if valid[ip] == 0:
                                plt.plot(pt[0, ip], pt[1, ip], 'v',
                                         color=colors[body['id'] % 7], markersize=3,)

                    if ip == 1:  # nose
                        # show id
                        plt.text(pt[0, ip], pt[1, ip]-100, '{0}'.format(body['id']),
                                 color=colors[body['id'] % 7],
                                 backgroundcolor='gray', fontsize='medium')
                # infer box
                temp_a = pt[0, :] < im.shape[1]
                temp_b = pt[0, :] >= 0
                x_mask = temp_a & temp_b
                # if not any(x_mask):
                #     continue
                temp_a = pt[1, :] < im.shape[0]
                temp_b = pt[1, :] >= 0
                y_mask = temp_a & temp_b
                # if not any(y_mask):
                #     continue
                mask = x_mask & y_mask
                if not any(mask):
                    continue
                x1 = min(min(pt[0, mask]), x1)   # only consider reachable points
                x2 = max(max(pt[0, mask]), x2)
                y1 = min(min(pt[1, mask]), y1)
                y2 = max(max(pt[1, mask]), y2)
                # plt.show()

        x1 = max(0, x1-box_inflate)
        x2 = min(im.shape[1], x2+box_inflate)
        y1 = max(0, y1-box_inflate)
        y2 = min(im.shape[0], y2+box_inflate)
        rect = patches.Rectangle(
            (x1, y1), (x2-x1+1), (y2-y1+1),
            linewidth=3, edgecolor='w', facecolor='none')
        currentAxis.add_patch(rect)

    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)

    plt.show()

a = 1
