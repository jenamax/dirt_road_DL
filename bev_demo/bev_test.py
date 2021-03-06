import cv2
from bev_transform import *
import numpy as np
import time
import yaml

def mask_preproc(mask):
    mask = bev.transform(mask)  # transform img to bird view perspective
    cv2.imwrite("bev.png", mask)
    mask = cv2.resize(mask, (width_compressed, height_compressed))  # compress

    return np.asarray(mask, dtype=np.uint8)

def get_lane_points(labels):
    lane_x = []
    lane_y = []
    weights = []
    for i in range(0, labels.shape[0], 20):  # write all lane points to array
        for j in range(0, labels.shape[1], 2):
            lane_x.append(i)
            lane_y.append(j)
            # assign weight of current point as brightness of corresponding
            # pixel
            weights.append(labels[i][j] + 0.01)
    return lane_x, lane_y, weights

pixels_on_meter = 40  # pixels per meter in bird view after transform
outputWidth = 800  # width of bird view img
outputHeight = 1500  # height of bird view img
width_compressed = outputWidth // 2
height_compressed = outputHeight // 2

with open("./bev_params.yaml") as file:
    bev_params = yaml.load(file, Loader=yaml.FullLoader)
    A = bev_params["yaw"]
    B = bev_params["pitch"]
    tx = bev_params["tx"]
    ty = bev_params["ty"]
    tz = bev_params["tz"]

camera_info = {'pitch': B, 'yaw': A, 'roll': 0,
                'tx': tx, 'ty': ty, 'tz': tz,
                'output_h': 1000, 'output_w': 800, 'pix_per_meter': 40,
               'img_h': 1080, 'img_w': 1920
}
calib_yaml_path = "./front_6mm_intrinsics.yaml"

bev = BEV(camera_info=camera_info, calib_yaml_path=calib_yaml_path)

img = cv2.imread("hough_img.png")
img = cv2.GaussianBlur(img, (11, 11), 0)

img = mask_preproc(img)

# get lane points coordinates and weights
lane_x, lane_y, weights = get_lane_points(img)
lane_x = (-np.asarray(lane_x) + img.shape[0]) / pixels_on_meter  # transform from img coordinates to car coordinates
lane_y = (-np.asarray(lane_y) + img.shape[1] / 2) / pixels_on_meter
weights = np.asarray(weights)

# fit the center points (in car coordinates) to polynom of 3rd degree
polynom_lane = np.polyfit(lane_x, lane_y, 3, w=weights)

# lane curvature calculation
x0 = 3.8
der_lane = np.array(
    [3 * polynom_lane[0], 2 * polynom_lane[1], polynom_lane[2]])
der_lane2 = np.array([6 * polynom_lane[0], 2 * polynom_lane[1]])
curv = int((1 + polynom_val(x0, der_lane) ** 2) ** (1.5) / abs(
    polynom_val(x0, der_lane2)))

# compute tangent line equation on the border of blind zone
tangent_line = np.array([polynom_val(x0, der_lane),
                         polynom_val(x0, polynom_lane) - x0 * polynom_val(
                             x0,
                             der_lane)])

pts_lane = []
lane_params_msg = CameraLaneLine()
# write points of the lane polynom to array (for visualization)
for x in np.arange(0.5, 38, 0.5):
    y = polynom_val(x, polynom_lane)
    p = Point3D()
    p.x = x
    p.y = y
    p.z = 0
    lane_params_msg.curve_camera_point_set.append(p)
    # transform from car coordinates to img coordinates
    pts_lane.append(
        (int(img.shape[1] / 2 - y * pixels_on_meter[0]),
         int(img.shape[0] - x * pixels_on_meter[1])))
pts_lane = np.array(pts_lane)

# get car shift value (it is equal to f(0) where f(x) - centerline
# polynom in car coordinates)
shift = tangent_line[-1]
angle = np.arctan(tangent_line[0])

vis = np.copy(img)
font = cv2.FONT_HERSHEY_SIMPLEX  # write the shift value on lane img
cv2.putText(vis, "%.2f" % shift, (2, 15), font, 0.5, 127, 1, cv2.LINE_AA)

# draw lane polynom (for visualisation)
cv2.polylines(vis, [pts_lane], False, 200)

cv2.imwrite("lane.png", vis)
