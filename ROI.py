from pathlib import Path

import cv2
import numpy as np

def compensate_ego_motion(prev_frame, curr_frame):
    # ORB 特征提取
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:500]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # RANSAC 估计仿射变换矩阵或透视变换
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return prev_frame

    h, w = curr_frame.shape[:2]
    warped_prev = cv2.warpPerspective(prev_frame, H, (w, h))
    return warped_prev


def extract_roi(prev_frame, curr_frame, threshold=40, min_area=800):
    # ego-motion 补偿
    aligned_prev = compensate_ego_motion(prev_frame, curr_frame)

    # 差分与前处理
    gray_prev = cv2.cvtColor(aligned_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_curr, gray_prev)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 形态学操作去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)

    # 轮廓检测
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = curr_frame[y:y+h, x:x+w]
            rois.append({'coords': (x, y, w, h), 'img': roi})
    return rois


# def extract_roi(frame_prev, frame_curr, threshold=25, min_area=200):
#     # gray
#     gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
#     gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
#
#     # diff
#     diff = cv2.absdiff(gray_curr, gray_prev)
#
#     # threshold
#     _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
#
#     # morphology
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
#
#     # find ROIs
#     contours, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     rois = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) > min_area:
#             x, y, w, h = cv2.boundingRect(cnt)
#             roi = frame_curr[y:y + h, x:x + w]
#             rois.append({'coords': (x, y, w, h), 'roi_img': roi})
#
#     return rois

# base_dir = "C:/Users/nouveau/Downloads/data_object_image_2/testing/image_2/"
# frame_prev = cv2.imread(base_dir + '000001.png')
# frame_curr = cv2.imread(base_dir + '000002.png')
# rois = extract_roi(frame_prev, frame_curr)
#
# for roi in rois:
#     x, y, w, h = roi['coords']
#     cv2.rectangle(frame_curr, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow('prev', frame_prev)
# cv2.imshow('ROIs', frame_curr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()