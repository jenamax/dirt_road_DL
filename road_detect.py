import time

import cv2
import numpy as np
import torch
from bev_transform import *

pix_on_meter = 16


# get set of road points from the segmentation mask
def get_road_points(mask):
    road_x = []
    road_y = []
    weights = []
    for i in range(0, mask.shape[0], 20):  # write all lane points to array
        for j in range(0, mask.shape[1], 2):
            road_x.append(i)
            road_y.append(j)
            # assign weight of current point as brightness of corresponding
            # pixel
            weights.append(mask[i][j] + 0.01)
    return road_x, road_y, weights


# get left and right lane points from mask and road center polynomial
def get_lanes_points(mask, road_center):
    lane1_x = []
    lane1_y = []

    lane2_x = []
    lane2_y = []

    for i in range(0, mask.shape[0], 20):  # write all lane points to array
        for j in range(0, mask.shape[1], 2):
            x = i
            y = road_center[0] * x**3 + road_center[1] * x**2 + road_center[2] * x + road_center[3]
            if mask[i][j] != 0: # chkeck that point is on road
                if y < j: # check whether it on the left or on the right from center
                   lane1_x.append(i)
                   lane1_y.append(j)
                else:
                    lane2_x.append(i)
                    lane2_y.append(j)
        
    return lane1_x, lane1_y, lane2_x, lane2_y

def get_polynom_points(poly, xlim): # get set of points by polynom equatin (used for visualization)
    pts = []
    for i in range(0, xlim):
        x = i
        y = poly[0] * x**3 + poly[1] * x**2 + poly[2] * x + poly[3]
        pts.append((y, x))  
    pts = np.array(pts, dtype=np.int32)
    return pts

def predict_on_img(model, img): # perform image segmentation
    image = torch.from_numpy(np.asarray(img.transpose(2, 0, 1), dtype=np.float32))

    x_tensor = image.to(device).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    return pr_mask


# load model
device = torch.device("cuda")
model = torch.load('./best_model.pth')

# read testing video and initialize video writer for result visualization
cap = cv2.VideoCapture("vid.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (512, 256))

while cap.isOpened():
    try:
        ret, frame = cap.read()
        s = time.time()
        img = cv2.resize(frame, (512, 256)) # resize image to model input shape

        res = predict_on_img(model, img) * 255 # segment and conver to int
        res = np.asarray(res, dtype=np.uint8)

        # detect road center
        mask_bev, transform_matrix = bird_eye_transform(res)
        road_x, road_y, weights = get_road_points(mask_bev)
        polynom_road = np.polyfit(road_x, road_y, 3, w=weights)
        pts_road = get_polynom_points(polynom_road, mask_bev.shape[0])
        cv2.polylines(mask_bev, [pts_road], False, 200)

        # estimate car shift relative to road center
        shift = (mask_bev.shape[0]**3 * polynom_road[0] + mask_bev.shape[0]**2 * polynom_road[1] 
        	+ mask_bev.shape[0] * polynom_road[2] + polynom_road[3] - mask_bev.shape[1] / 2) / pix_on_meter

        # get left and right lanes points
        lane1_x, lane1_y, lane2_x, lane2_y = get_lanes_points(mask_bev, polynom_road)

        # estimate lane center lines
        lane1 = np.polyfit(lane1_x, lane1_y, 3)
        pts_lane1 = get_polynom_points(lane1, mask_bev.shape[0])
        lane2 = np.polyfit(lane2_x, lane2_y, 3)
        pts_lane2 = get_polynom_points(lane2, mask_bev.shape[0])

        # estimate lane width
        lane1_w = (mask_bev.shape[0]**3 * lane1[0] + mask_bev.shape[0]**2 * lane1[1] 
        	+ mask_bev.shape[0] * lane1[2] + lane1[3] - mask_bev.shape[1] / 2) / pix_on_meter
        lane2_w = (mask_bev.shape[0]**3 * lane2[0] + mask_bev.shape[0]**2 * lane2[1] 
        	+ mask_bev.shape[0] * lane2[2] + lane2[3] - mask_bev.shape[1] / 2) / pix_on_meter
        width = abs(lane1_w - lane2_w) / pix_on_meter


        # convert masks to RGB for visualization convenience
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        mask_bev = cv2.cvtColor(mask_bev, cv2.COLOR_GRAY2RGB)
        
        # visuzlization staff (draw center lines, overlay mask on image)
        cv2.polylines(mask_bev, [pts_lane1], False, (255, 0, 0))
        cv2.polylines(mask_bev, [pts_lane2], False, (0, 0, 255))
        alpha = 0.7
        vis = np.copy(img)
        mask_bev = cv2.warpPerspective(mask_bev, np.linalg.inv(transform_matrix),  (img.shape[1], img.shape[0]))
        cv2.addWeighted(img, alpha, mask_bev, 1 - alpha, 0, vis)

        # draw shift value on image, mark wheather road has one or two lanes
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(vis, "%.2f" % shift, (6, 22), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        if width > 3.5:
            cv2.putText(vis, "Two lanes", (6, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "One lane", (6, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        if ret == True:
            # cv2.imshow("seg", vis)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            out.write(vis)
            cv2.imwrite("vis.png", vis)
        else:
            break
        print(time.time() - s)
    except KeyboardInterrupt:
        break
cap.release()

out.release()

cv2.destroyAllWindows()
