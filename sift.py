import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd


base_path = "E:\textbackslash\textbackslash 24_03_2022_run_102_C001H001S0001\textbackslash\textbackslash 1_200_S0001"

file_path = base_path

save_folder_name = base_path.split("\textbackslash\textbackslash")[3].split("_")[-2]
save_folder_location = os.path.join("C:\Thesis Calculations\textbackslash\textbackslash DIC\textbackslash\textbackslash", save_folder_name)

if not os.path.exists(save_folder_location):
    os.makedirs(save_folder_location)

tiff_files = [f for f in os.listdir(file_path) 
                        if f.endswith(".tif") or f.endswith(".tiff")]

\textcolor{green!45!red}{"""}
\textcolor{green!45!red}{1 pixel = 0.26 mm}
\textcolor{green!45!red}{1 mm = 3.846 pixel, 75 mm = 288.45 pixel}

\textcolor{green!45!red}{"""}

def top_patch(image):
    midline = 223           # 3.846*55 = 211.53, 211.53 + 11 = 222.53 = 223
    ul = midline - 50
    ll = midline + 50
    horizontal_Center = 177
    hl = horizontal_Center - 50
    hr = horizontal_Center + 50
    image_top_patch = image[ul:ll, hl:hr]    \textcolor{green!65!blue}{#80:280}
    return image_top_patch

def mid_patch(image):
    midline = 511           \textcolor{green!65!blue}{# 223 + 288.45 = 511}
    ul = midline - 50
    ll = midline + 50
    horizontal_Center = 177
    hl = horizontal_Center - 50
    hr = horizontal_Center + 50
    image_mid_patch = image[ul:ll, hl:hr] 
    return image_mid_patch

def bottom_patch(image):
    midline = 799           \textcolor{green!65!blue}{#  511 + 288.45 = 799.45}
    ul = midline - 50
    ll = midline + 50
    horizontal_Center = 177
    hl = horizontal_Center - 50
    hr = horizontal_Center + 50
    image_bottom_patch = image[ul:ll, hl:hr] 
    return image_bottom_patch

def SIFTdetector(image_patch_1, image_patch_2):
    gray_img1 = cv2.cvtColor(image_patch_1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image_patch_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 10000, 
                contrastThreshold = 0.01, edgeThreshold = 10, sigma = 1.6)

    \textcolor{green!65!blue}{# find the keypoints and descriptors with sift}
    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    img1_kp = cv2.drawKeypoints(gray_img1, kp1, 0, (0, 0, 255))
    img2_kp = cv2.drawKeypoints(gray_img2, kp2, 0, (0, 0, 255))

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k = 2)
    good_matches = []
    for (m, n) in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    \textcolor{green!65!blue}{# match_image = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None)}

    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)

    points1 = np.float32([kp1[good_match.queryIdx].pt 
                for good_match in good_matches])
    points2 = np.float32([kp2[good_match.trainIdx].pt 
                for good_match in good_matches])

    diff = pd.DataFrame(points1 - points2)

    z = np.abs(stats.zscore(diff[0]))
    z1 = np.abs(stats.zscore(diff[1]))

    point_z = np.where(z < 1)
    point_z1 = np.where(z < 0.7)
    indices_final = point_z and point_z1

    points1_final=np.empty((np.size(indices_final),2))
    points2_final=np.empty((np.size(indices_final),2))
    k=0

    for pts in indices_final[0]:
        \textcolor{green!65!blue}{#  print(pts)}
        points1_final[k][0]=points1[pts][0]
        points1_final[k][1]=points1[pts][1]
        points2_final[k][0]=points2[pts][0]
        points2_final[k][1]=points2[pts][1]
        k=k+1

    points1_final=np.float32(points1_final)
    points2_final=np.float32(points2_final)

    x1 = points1_final[:,0]
    y1 = points1_final[:,1]
    x2 = points2_final[:,0]
    y2 = points2_final[:,1]
    
    X = x1 - x2
    Y = y1 - y2

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    mean_x = mean_x * (-1)
    mean_y = mean_y * (-1)

    return mean_x, mean_y

def Patch_Creator(image1, image2):
    img1_t = top_patch(image1)
    img2_t = top_patch(image2)

    img1_m = mid_patch(image1)
    img2_m = mid_patch(image2)

    img1_b = bottom_patch(image1)
    img2_b = bottom_patch(image2)

    mean_x_t, mean_y_t = SIFTdetector(img1_t, img2_t)
    mean_x_m, mean_y_m = SIFTdetector(img1_m, img2_m)
    mean_x_b, mean_y_b = SIFTdetector(img1_b, img2_b)

    return mean_x_t, mean_y_t, mean_x_m, mean_y_m, mean_x_b, mean_y_b

x_disp_cumu_top = []
y_disp_cumu_top = []
x_disp_cumu_mid = []
y_disp_cumu_mid = []
x_disp_cumu_btm = []
y_disp_cumu_btm = []

for i in range(len(tiff_files) - 1):
    \textcolor{green!65!blue}{# Load two consecutive TIFF images}
    img1 = cv2.imread(os.path.join(file_path, tiff_files[0]))
    img2 = cv2.imread(os.path.join(file_path, tiff_files[i]))

    x_disp_inst_top, y_disp_inst_top, x_disp_inst_mid, y_disp_inst_mid, 
        x_disp_inst_btm, y_disp_inst_btm = Patch_Creator(img1, img2)

    x_disp_cumu_top.append(x_disp_inst_top)
    y_disp_cumu_top.append(y_disp_inst_top)

    x_disp_cumu_mid.append(x_disp_inst_mid)
    y_disp_cumu_mid.append(y_disp_inst_mid)

    x_disp_cumu_btm.append(x_disp_inst_btm)
    y_disp_cumu_btm.append(y_disp_inst_btm)

    

X_disp_top = [i * 0.2267 for i in x_disp_cumu_top]
Y_disp_top = [i * 0.2267 for i in y_disp_cumu_top]

X_disp_mid = [i * 0.2267 for i in x_disp_cumu_mid]
Y_disp_mid = [i * 0.2267 for i in y_disp_cumu_mid]

X_disp_btm = [i * 0.2267 for i in x_disp_cumu_btm]
Y_disp_btm = [i * 0.2267 for i in y_disp_cumu_btm]

t_step = 0.0001
L = len(x_disp_cumu_top)
t = []
t_0 = 0
t.append(t_0)
for i in range(len(x_disp_cumu_top) - 1):
     t_0 = t_0 + t_step
     t.append(t_0)

\textcolor{green!65!blue}{# print(len(t))}

df_tuple = ('pixel_disp_', save_folder_name, ".csv")
df_name = "".join(df_tuple)
df_path = os.path.join(save_folder_location, df_name)
df = pd.DataFrame(list(zip(t, x_disp_cumu_top, y_disp_cumu_top, 
x_disp_cumu_mid, y_disp_cumu_mid, x_disp_cumu_btm, y_disp_cumu_btm)), 
columns =['Time (s)', 'X_top (pix))', 'Y_top (pix)', 'X_mid (pix)', 
'Y_mid (pix)', 'X_btm (pix)', 'Y_btm (pix)'])
df.to_csv(df_path, index=False)
print('Done')


VD_plt_tuple_top = ('Cumulative_Vertical_Displacement_top_patch', 
    save_folder_name, ".jpg")
VD_plt_name_top = "".join(VD_plt_tuple_top)
plt_V_top = os.path.join(save_folder_location, VD_plt_name_top)
plt.figure(num = 1)
plt.title('Vertical Displacement Top Patch')
plt.xlabel('Sec')
plt.ylabel('mm')
plt.plot(t, Y_disp_top)
plt.savefig(plt_V_top, dpi = 300)

HD_plt_tuple_top = ('Cumulative_Horizontal_Displacement_top_patch', 
    save_folder_name, ".jpg")
HD_plt_name_top = "".join(HD_plt_tuple_top)
plt_H_top = os.path.join(save_folder_location, HD_plt_name_top)
plt.figure(num = 2)
plt.title('Horizontal Displacement Top Patch')
plt.xlabel('Sec')
plt.ylabel('mm')
plt.plot(t, X_disp_top)
plt.savefig(plt_H_top, dpi = 300)

VD_plt_tuple_mid = ('Cumulative_Vertical_Displacement_mid_patch', 
    save_folder_name, ".jpg")
VD_plt_name_mid = "".join(VD_plt_tuple_mid)
plt_V_mid = os.path.join(save_folder_location, VD_plt_name_mid)
plt.figure(num = 3)
plt.title('Vertical Displacement Mid Patch')
plt.xlabel('Sec')
plt.ylabel('mm')
plt.plot(t, Y_disp_mid)
plt.savefig(plt_V_mid, dpi = 300)

HD_plt_tuple_mid = ('Cumulative_Horizontal_Displacement_mid_patch', 
    save_folder_name, ".jpg")
HD_plt_name_mid = "".join(HD_plt_tuple_mid)
plt_H_mid = os.path.join(save_folder_location, HD_plt_name_mid)
plt.figure(num = 4)
plt.title('Horizontal Displacement Mid Patch')
plt.xlabel('Sec')
plt.ylabel('mm')
plt.plot(t, X_disp_mid)
plt.savefig(plt_H_mid, dpi = 300)

VD_plt_tuple_btm = ('Cumulative_Vertical_Displacement_btm_patch', 
    save_folder_name, ".jpg")
VD_plt_name_btm = "".join(VD_plt_tuple_btm)
plt_V_btm = os.path.join(save_folder_location, VD_plt_name_btm)
plt.figure(num = 5)
plt.title('Vertical Displacement Bottom Patch')
plt.xlabel('Sec')
plt.ylabel('mm')
plt.plot(t, Y_disp_btm)
plt.savefig(plt_V_btm, dpi = 300)

HD_plt_tuple_btm = ('Cumulative_Horizontal_Displacement_btm_patch', 
    save_folder_name, ".jpg")
HD_plt_name_btm = "".join(HD_plt_tuple_btm)
plt_H_btm = os.path.join(save_folder_location, HD_plt_name_btm)
plt.figure(num = 6)
plt.title('Horizontal Displacement Bottom Patch')
plt.xlabel('Sec')
plt.ylabel('mm')
plt.plot(t, X_disp_btm)
plt.savefig(plt_H_btm, dpi = 300)

\textcolor{green!65!blue}{# print('ALL GOOD')}
df_tuple = ('disp_', save_folder_name, ".csv")
df_name = "".join(df_tuple)
df_path = os.path.join(save_folder_location, df_name)
df = pd.DataFrame(list(zip(t, X_disp_top, Y_disp_top, X_disp_mid, 
Y_disp_mid, X_disp_btm, Y_disp_btm)), columns =['Time (s)', 'X_top (mm)', 
'Y_top (mm)', 'X_mid (mm)', 'Y_mid (mm)', 'X_btm (mm)', 'Y_btm (mm)'])
df.to_csv(df_path, index=False)
print('Done')
\textcolor{green!65!blue}{# plt.figure()}
\textcolor{green!65!blue}{# plt.plot(t, y_disp_cumu)}
