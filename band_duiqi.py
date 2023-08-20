# coding=utf-8
import cv2
import numpy as np
import math


def drawMatches(img1, img2, good_matches):
    img_out = np.hstack((img1, img2))
    for match in good_matches:
        pt1 = (int(match[0]), int(match[1]))
        pt2 = (int(match[2] + img1.shape[1]), int(match[3]))
        cv2.circle(img_out, pt1, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img_out, pt2, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img_out, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return img_out


def extractPointFromMatches(good_matches):
    keypoints1 = []
    keypoints2 = []
    for match in good_matches:
        keypoints1.append([match[0], match[1]])
        keypoints2.append([match[2], match[3]])
    return keypoints1, keypoints2


def calcDis(matches):
    distences = []
    for match in matches:
        x1 = match[0]
        y1 = match[1]
        x2 = match[2]
        y2 = match[3]
        dis = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        distences.append(dis)
    return distences


def dataFilter(matches, ratio):
    # 由于这里认为不同影像间仅存在平移关系
    # 因此各匹配点对之间的连线距离应该是相等且斜率相等的
    # 筛选也是依据这两个条件进行的
    # 如果两个图像间存在别的对应关系，这个筛选条件就失效了

    # 求平均值
    ave_dis = 0
    ave_k = 0
    for match in matches:
        x1 = match[0]
        y1 = match[1]
        x2 = match[2]
        y2 = match[3]
        dis = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        k = (y2 - y1) / (x2 - x1)
        ave_dis = ave_dis + dis
        ave_k = ave_k + k
    ave_dis = ave_dis / matches.__len__()
    ave_k = ave_k / matches.__len__()

    # 求标准差
    stdv_dis = 0
    stdv_k = 0
    for match in matches:
        x1 = match[0]
        y1 = match[1]
        x2 = match[2]
        y2 = match[3]
        dis = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        k = (y2 - y1) / (x2 - x1)
        stdv_dis = stdv_dis + math.pow(dis - ave_dis, 2)
        stdv_k = stdv_k + math.pow(k - ave_k, 2)
    stdv_dis = math.sqrt(stdv_dis / matches.__len__())
    stdv_k = math.sqrt(stdv_k / matches.__len__())

    # 求置信范围
    range_top = ave_dis + ratio * stdv_dis
    range_bottom = ave_dis - ratio * stdv_dis
    range_k_top = ave_k + ratio * stdv_k
    range_k_bottom = ave_k - ratio * stdv_k

    print("\nave_dis " + ave_dis.__str__())
    print("range_top " + range_top.__str__())
    print("range_bottom " + range_bottom.__str__())
    print("ave_k " + ave_k.__str__())
    print("range_top_k " + range_k_top.__str__())
    print("range_bottom_k " + range_k_bottom.__str__())

    # 筛选
    good_matches = []
    for match in matches:
        x1 = match[0]
        y1 = match[1]
        x2 = match[2]
        y2 = match[3]
        dis = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        k = abs((y2 - y1) / (x2 - x1))
        if dis < range_top and dis > range_bottom and k < range_k_top and k > range_k_bottom:
            good_matches.append(match)
    return good_matches


# FLANN+SIFT
# 由于自定义的筛选条件比较严格，开启筛选后很有可能候选点一个都不满足，但其实匹配效果还是不错的，所以默认关闭
# 而且，由于后面在计算单应矩阵时，本身也会用RANSAC筛选，所以这里可以不过滤
def FLANN_SIFT(img1, img2, flag=False):
    # 新建SIFT对象，参数默认
    sift = cv2.xfeatures2d_SIFT.create()
    # 调用函数进行SIFT提取
    kp1, des1 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img1, None)
    kp2, des2 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img2, None)

    if len(kp1) == 0 or len(kp2) == 0:
        print("No enough keypoints.")
        return
    else:
        print("\nkp1 size:" + len(kp1).__str__() + "," + "kp2 size:" + len(kp2).__str__())

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    good_kps1 = []
    good_kps2 = []

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    if good_matches.__len__() == 0:
        print("No enough good matches.")
        cv2.drawKeypoints(img1, kp1, img1)
        cv2.drawKeypoints(img2, kp2, img2)
        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)
        return
    else:
        good_out = []
        good_out_kp1 = []
        good_out_kp2 = []
        print("good matches:" + good_matches.__len__().__str__())
        for i in range(good_kps1.__len__()):
            good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
            good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
            good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])

        if flag == True:
            # 筛选匹配
            good_out = dataFilter(good_out, 3)
            good_out_kp1, good_out_kp2 = extractPointFromMatches(good_out)

    img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img3 = drawMatches(img1_show, img2_show, good_out)
    cv2.imshow('match result', img3)
    cv2.waitKey(0)
    return good_out_kp1, good_out_kp2, good_out


# 分别读取RGB三个波段数据
band_b = cv2.imread('data/01000v/b.jpg', cv2.IMREAD_GRAYSCALE)
band_g = cv2.imread('data/01000v/g.jpg', cv2.IMREAD_GRAYSCALE)
band_r = cv2.imread('data/01000v/r.jpg', cv2.IMREAD_GRAYSCALE)

# 分别对B-G、B-R进行匹配
kp1, kp2, matches1 = FLANN_SIFT(band_b, band_g)
kp3, kp4, matches2 = FLANN_SIFT(band_b, band_r)

# 依据匹配的特征点求解不同波段间的变换关系
# 同时这里要注意对不同匹配结果的应对措施
# 如果一对同名点都没有匹配到，那就直接原图搬过去
# 如果匹配到小于4对同名点，无法计算单应矩阵，那就通过求解偏移的平均值手动构造
# 如果正常匹配到很多对点，利用OpenCV求解单应矩阵
if kp1 is None:
    homo1 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
elif kp1.__len__() < 4:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    for i in range(kp1.__len__()):
        x1 = x1 + kp1[i][0]
        y1 = y1 + kp1[i][1]
        x2 = x2 + kp2[i][0]
        y2 = y2 + kp2[i][1]
    x1 = x1 / kp1.__len__()
    x2 = x2 / kp1.__len__()
    y1 = y1 / kp1.__len__()
    y2 = y2 / kp1.__len__()
    dx = x2 - x1
    dy = y2 - y1

    homo1 = np.array([[1, 0, dx],
                      [0, 1, dy],
                      [0, 0, 1]])
else:
    homo1, mask1 = cv2.findHomography(np.array(kp2), np.array(kp1), cv2.RANSAC)
if kp3 is None:
    homo2 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
elif kp3.__len__() < 4:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    for i in range(kp3.__len__()):
        x1 = x1 + kp3[i][0]
        y1 = y1 + kp3[i][1]
        x2 = x2 + kp4[i][0]
        y2 = y2 + kp4[i][1]
    x1 = x1 / kp1.__len__()
    x2 = x2 / kp1.__len__()
    y1 = y1 / kp1.__len__()
    y2 = y2 / kp1.__len__()
    dx = x2 - x1
    dy = y2 - y1

    homo2 = np.array([[1, 0, dx],
                      [0, 1, dy],
                      [0, 0, 1]])
else:
    homo2, mask2 = cv2.findHomography(np.array(kp4), np.array(kp3), cv2.RANSAC)
print("Homography between B and G band")
print(homo1)
print("Homography between B and R band")
print(homo2)

# 依据求得的单应矩阵对波段进行重采
resampled_band_g = cv2.warpPerspective(band_g, homo1, (band_g.shape[1], band_g.shape[0]))
resampled_band_r = cv2.warpPerspective(band_r, homo2, (band_r.shape[1], band_r.shape[0]))

# 将不同波段合并成新的彩色图像
img = np.zeros([band_b.shape[0], band_b.shape[1], 3], np.uint8)
img[:, :, 0] = band_b
img[:, :, 1] = resampled_band_g
img[:, :, 2] = resampled_band_r

# 直接读取的RGB波段叠加形成的彩色影像
img2 = np.zeros([band_b.shape[0], band_b.shape[1], 3], np.uint8)
img2[:, :, 0] = band_b
img2[:, :, 1] = band_g
img2[:, :, 2] = band_r

cv2.imshow("Resample band data", img)
cv2.imshow("Overlay band data", img2)
cv2.waitKey(0)

