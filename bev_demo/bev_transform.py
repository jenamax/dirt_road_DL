import sys

import numpy as np
import yaml
import cv2
import os


def get_RX(pitch_angle):
    pitch_angle = (np.pi / 180) * pitch_angle
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(pitch_angle), -np.sin(pitch_angle), 0],
        [0, np.sin(pitch_angle), np.cos(pitch_angle), 0],
        [0, 0, 0, 1]
    ])


def get_RY(yaw_angle):
    yaw_angle = (np.pi / 180) * yaw_angle
    return np.array([
        [np.cos(yaw_angle), 0, np.sin(yaw_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(yaw_angle), 0, np.cos(yaw_angle), 0],
        [0, 0, 0, 1]
    ])


def get_RZ(roll_angle):
    roll_angle = (np.pi / 180) * roll_angle
    return np.array([
        [np.cos(roll_angle), -np.sin(roll_angle), 0, 0],
        [np.sin(roll_angle), np.cos(roll_angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def get_T(vtx, vty, vtz):
    return np.array([
        [1, 0, 0, vtx],
        [0, 1, 0, vty],
        [0, 0, 1, vtz],
        [0, 0, 0, 1]])


def calculate_BEV_H(calib_params, camera_info=None, pix_per_meter=20):
    if camera_info is None:
        camera_info = cfg.camera_info[cfg.camera_name]
    output_w = camera_info['output_w']
    output_h = camera_info['output_h']

    RX = get_RX(camera_info['pitch'])
    RY = get_RY(camera_info['yaw'])
    RZ = get_RZ(camera_info['roll'])
    T = get_T(camera_info['tx'],
              camera_info['ty'],
              camera_info['tz'])

    camera2xyz = get_RX(90) @ get_RZ(180)
    camera2loco = camera2xyz @ RZ @ RY @ RX @ T

    ex_loco = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    camera2loco = ex_loco @ camera2loco

    R = camera2loco[:3, :3]
    T = camera2loco[:3, 3]

    K = np.asarray(calib_params['K']).reshape((3, 3))
    H = np.zeros((3, 3))
    H[:, :2] = (K @ R.T)[:3, :2]
    H[:, 2] = -K @ R.T @ T
    H_inv = np.linalg.inv(H)
    image2ground = H_inv

    meters_to_pix = np.array([
        [0, -pix_per_meter, output_w * 0.5],
        [-pix_per_meter, 0, output_h],
        [0, 0, 1]
    ])
    image2ground = meters_to_pix @ image2ground

    return image2ground


def get_BEV_H(camera_info, calib_yaml_path):
    # if calib_yaml_path is None:
    #     calib_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                                    cfg.calib_file_path[cfg.camera_name])
    params_to_parse = ['K',
                       'D',
                       'P',
                       'R']
    calib_matrices = {}

    with open(calib_yaml_path) as file:
        calibration_params = yaml.load(file, Loader=yaml.FullLoader)
        calib_matrices['width'] = int(calibration_params['width'])
        calib_matrices['height'] = int(calibration_params['height'])

        calib_matrices['K'] = np.asarray(calibration_params['K']).reshape((3, 3))
        calib_matrices['D'] = np.asarray(calibration_params['D'])
        calib_matrices['P'] = np.asarray(calibration_params['P']).reshape((3, 4))
        calib_matrices['R'] = np.asarray(calibration_params['R']).reshape((3, 3))

    H = calculate_BEV_H(calib_matrices, camera_info, pix_per_meter=camera_info['pix_per_meter'])
    return H, calib_matrices


class BEV(object):
    def __init__(self, camera_info=None, calib_yaml_path=None):
        self.H, self.calib_matrices = get_BEV_H(camera_info, calib_yaml_path)
        print(self.calib_matrices)
        self.inv_H = np.linalg.inv(self.H)
        self.pixels_per_meter = camera_info['pix_per_meter']
        self.output_w = camera_info['output_w']
        self.output_h = camera_info['output_h']
        self.blind_zone = camera_info['tz']

        self.scale_x = float(camera_info['img_w']) / self.calib_matrices['width']
        self.scale_y = float(camera_info['img_w']) / self.calib_matrices['height']

        self.calib_matrices['K'][0, 0] *= self.scale_x
        self.calib_matrices['K'][1, 1] *= self.scale_y
        self.calib_matrices['K'][0, 2] *= self.scale_x
        self.calib_matrices['K'][1, 2] *= self.scale_y

        self.f_x = self.calib_matrices['K'][0, 0]
        self.f_y = self.calib_matrices['K'][1, 1]

    def transform(self, img):
        transformed_img = cv2.warpPerspective(img, self.H, (self.output_w,
                                                            self.output_h))
        return transformed_img

    def calculate_dist_meters(self, p_meters):
        dists = np.sqrt(p_meters[:, 0] * p_meters[:, 0] + p_meters[:, 1] * p_meters[:, 1])
        return dists

    def calculate_dist(self, points_bev):
        new_p_centered = np.array([self.output_w / 2, self.output_h]) - points_bev
        new_p_meters = new_p_centered / self.pixels_per_meter
        dists = self.calculate_dist_meters(new_p_meters)
        return dists

    def calculate_dist_bev(self, points):
        points_bev = self.points_to_bev(points)
        return self.calculate_dist(points_bev) + self.blind_zone

    def points_to_bev(self, points):
        points_ex = np.ones((points.shape[0], points.shape[1] + 1))
        points_ex[:, :2] = points
        new_p = self.H @ points_ex.T
        new_p = new_p.T
        new_p /= new_p[:, 2:]
        new_p = new_p[:, :2]

        return new_p

    def bev_to_points(self, points):
        points_ex = np.ones((points.shape[0], points.shape[1] + 1))
        points_ex[:, :2] = points
        new_p = self.inv_H @ points_ex.T
        new_p = new_p.T
        new_p /= new_p[:, 2:]
        new_p = new_p[:, :2]

        return new_p

    def pixels_to_meters(self, points):
        points_bev = self.points_to_bev(points)
        new_p_centered = np.array([self.output_w / 2, self.output_h]) - points_bev
        new_p_meters = new_p_centered / self.pixels_per_meter
        new_p_meters[:, 0] *= -1
        return new_p_meters

    def meters_to_pixels(self, points):
        new_p_pixels = self.pixels_per_meter * points
        new_p_uncentered = np.array([self.output_w / 2, self.output_h]) - new_p_pixels
        new_p_pixels = self.bev_to_points(new_p_uncentered)

        return new_p_pixels

    def get_height_in_pixels(self, height, distance):
        return self.f_y * height / distance

    def __call__(self, img):
        return self.transform(img)
