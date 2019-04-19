import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'

# For camera projection (with distortion)
import panutils
import cv2


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

COLORS = np.array([
    [255, 0, 0],        # 0
    [255, 85, 0],       # 1
    [255, 170, 0],      # 2
    [255, 255, 0],      # 3
    [170, 255, 0],      # 4
    [85, 255, 0],       # 5
    [0, 255, 0],        # 6
    [0, 255, 85],       # 7
    [0, 255, 170],      # 8
    [0, 255, 255],      # 9
    [0, 170, 255],      # 10
    [0, 85, 255],       # 11
    [0, 0, 255],        # 12
    [85, 0, 255],       # 13
    [170, 0, 255],      # 14
    [255, 0, 255],      # 15
    [255, 0, 170],      # 16
    [255, 0, 85],       # 17
    [255, 0, 0]         # 18
])
COLORS = COLORS / 255.

# Edges between joints in the body skeleton
body_edges = np.array([
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 9),
    # (1, 15),
    # (1, 17),
    (2, 6),
    (2, 12),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (12, 13),
    (13, 14),
    # (15, 16),
    # (17, 18),  # 18 connections in total
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
IMAGE_SIZE = [256, 256]  # w, h
SPEC_BODY = 44
cam_list = [
    (0, 26),
    (0, 15),
    (0, 10),
    # (0, 15),
    # (0, 20),
    # (0, 25),
    (0, 30),
]
for entry in cam_list:
    # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera
    cam = cameras[entry]

    # Load the corresponding HD image
    image_path = hd_img_path+'{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'], cam['node'], hd_idx)
    im = plt.imread(image_path)
    data_numpy = cv2.imread(
        image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

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
            # if body['id'] in [43, 46]:
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
                for edge_id, edge in enumerate(body_edges):
                    if valid[edge[0]] and valid[edge[1]]:
                        plt.plot(pt[0, edge], pt[1, edge], color=colors[body['id'] % 7])
                        # plt.plot(pt[0, edge], pt[1, edge], color=COLORS[edge_id])

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

                if body['id'] == SPEC_BODY or SPEC_BODY == -1:
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
                    # a = 1

        x1 = max(0, x1-box_inflate)
        x2 = min(im.shape[1], x2+box_inflate)
        y1 = max(0, y1-box_inflate)
        y2 = min(im.shape[0], y2+box_inflate)
        rect = patches.Rectangle(
            (x1, y1), (x2-x1+1), (y2-y1+1),
            linewidth=3, edgecolor='w', facecolor='none')
        currentAxis.add_patch(rect)

        box = [x1, y1, x2, y2]
        center = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
        scale = np.array([(box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0])
        trans = get_affine_transform(center, scale, 0, IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans, (IMAGE_SIZE[0], IMAGE_SIZE[1]),
            flags=cv2.INTER_LINEAR)

        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 15))
        plt.imshow(input)

    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)

    plt.show()
    a = 1   # current view STOP

a = 1
