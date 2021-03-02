import time

import cv2
import numpy as np
import torch


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
        img = cv2.resize(frame, (512, 256))

        s = time.time()
        res = predict_on_img(model, img) * 255
        print(time.time() - s)
        res = np.asarray(res, dtype=np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        alpha = 0.7
        vis = np.copy(img)
        cv2.addWeighted(img, alpha, res, 1 - alpha, 0, vis)
        if ret == True:
            cv2.imshow("seg", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            out.write(vis)
        else:
            break
    except KeyboardInterrupt:
        break
cap.release()

out.release()

cv2.destroyAllWindows()
