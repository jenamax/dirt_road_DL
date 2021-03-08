import time

import cv2
import numpy as np
import torch
from bev_transform import *

pix_on_meter = 16

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


def predict_on_img(model, img):
    image = torch.from_numpy(np.asarray(img.transpose(2, 0, 1), dtype=np.float32))

    x_tensor = image.to(device).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    return pr_mask


device = torch.device("cuda")
model = torch.load('./best_model.pth')

cap = cv2.VideoCapture("vid.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (512, 256))

while cap.isOpened():
    try:
        ret, frame = cap.read()
        s = time.time()
        img = cv2.resize(frame, (512, 256))
        res = predict_on_img(model, img) * 255
        
        res = np.asarray(res, dtype=np.uint8)

        mask_bev = bird_eye_transform(res)
        lane_x, lane_y, weights = get_lane_points(mask_bev)
        polynom_lane = np.polyfit(lane_x, lane_y, 3, w=weights)
        pts_lane = []
        for i in range(0, mask_bev.shape[0]):
            x = i
            y = polynom_lane[0] * x**3 + polynom_lane[1] * x**2 + polynom_lane[2] * x + polynom_lane[3]
            pts_lane.append((y, x))  
        pts_lane = np.array(pts_lane, dtype=np.int32)
        cv2.polylines(mask_bev, [pts_lane], False, 200)
        shift = (polynom_lane[3] - mask_bev.shape[1] / 2) / pix_on_meter
        font = cv2.FONT_HERSHEY_SIMPLEX 
        

        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        mask_bev = cv2.cvtColor(mask_bev, cv2.COLOR_GRAY2RGB)
        cv2.putText(mask_bev, "%.2f" % shift, (6, 22), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        alpha = 0.7
        vis = np.copy(img)
        cv2.addWeighted(img, alpha, res, 1 - alpha, 0, vis)
        mask_bev = cv2.resize(mask_bev, (vis.shape[1], vis.shape[0]))
        vis = cv2.vconcat([vis, mask_bev])
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
