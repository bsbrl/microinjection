import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
from scipy.linalg import svd
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random

def find_match(img1, img2):
    # TO DO
    # Calculating sift for img1 and img2
    sift1 = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift1.detectAndCompute(img1,None)
    
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(img2,None)
    
    # NearestNeighbors similar to HW2
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    kp_required = neigh.kneighbors(des1,2,return_distance = True)
    df1 = pd.DataFrame(kp_required[0])
    df2 = pd.DataFrame(kp_required[1])
    df_all = pd.concat([df1, df2], axis=1)
    df_all.columns = list(['d1','d2','x','y'])
    df_all['ratio'] = df_all['d1']/df_all['d2']
    dfkp = df_all[df_all['ratio'] < .9]
    x1_kp = pd.Series(kp1)[dfkp.index.values]
    x1_kp = x1_kp.values
    x2_kp = pd.Series(kp2)[dfkp['x']]
    x2_kp = x2_kp.values
    x1 = [x1_kp[idx].pt for idx in range(0, len(x1_kp))]
    x2 = [x2_kp[idx].pt for idx in range(0, len(x2_kp))]
    pts1 = np.array(x1)
    pts2 = np.array(x2)
    
    return pts1, pts2


def compute_F(pts1, pts2):
    # TO DO
    total_iter = 100000;
    inlines = np.zeros((total_iter,1));
    F_matrix = [];
    iter_numb = 0;
    threshold = 0.01;
    rand_numb = len(pts1)
    while (iter_numb < total_iter):
        if iter_numb % 1000 == 0:
            print('Current iter number', iter_numb)
        flag = 0
        A_mat = []
        list_rand = random.sample(range(0, rand_numb), 8)
        for i in list_rand:
            A_temp = [pts1[i,0]*pts2[i,0], pts1[i,1]*pts2[i,0], pts2[i,0], pts1[i,0]*pts2[i,1], pts1[i,1]*pts2[i,1], pts2[i,1], pts1[i,0], pts1[i,1], 1.0]
            A_mat.append(A_temp) 
        A_mat = np.array(A_mat)
        fn = null_space(A_mat)[:,0]
        f_mat = fn.reshape((3,3))
        f_mat = f_mat/f_mat[2,2]
        [U,D,V] = svd(f_mat)
        D[2] = 0
        mod_f = U@np.diag(D) @ V
        for i in range(0, rand_numb):
            u = np.array([pts1[i,0],pts1[i,1],1])
            v = np.array([pts2[i,0],pts2[i,1],1])
            err = (v.T) @ mod_f @ u
            if(np.abs(err) < threshold):
                flag = flag + 1
        inlines[iter_numb] = flag
        F_matrix.append(mod_f)
        iter_numb = iter_numb + 1
    idx = np.argmax(inlines)
    F = F_matrix[idx]
    return F

def triangulation(P1, P2, pts1, pts2):
    # TO DO
    pts3D = []
    for i in range(0, np.shape(pts1)[0]):
        Upper_A = [[0, -1, pts1[i,1]], [1, 0, -pts1[i,0]], [-pts1[i,1], pts1[i,0], 0]] @ P1
        Lower_A = [[0, -1, pts2[i,1]], [1, 0, -pts2[i,0]], [-pts2[i,1], pts2[i,0], 0]] @ P2
        [U, S, V] = svd(np.vstack([Upper_A, Lower_A]))
        pts3D.append(V[0:3,3]/V[3,3])
    pts3D = np.array(pts3D)
    return pts3D

def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    XYZ = []
    for i in range(0,4):
        R = Rs[i]
        C = Cs[i]
        P_3D = pts3Ds[i]
        flag = 0
        for j in range(0, len(P_3D)):
            c_temp = R[2,:] @ ((P_3D[j,:] - C.T).T)
            if (c_temp > 0):
                flag = flag +1
        XYZ.append(flag)
    R = Rs[np.argmax(XYZ)]
    C = Cs[np.argmax(XYZ)]
    pts3D = pts3Ds[np.argmax(XYZ)]
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    rx_t = (C/np.linalg.norm(C)).T
    rz_tilde = np.array([[0],[0],[1]])
    rz = (rz_tilde - ((rz_tilde.T @ rx_t.T) * rx_t.T))/np.linalg.norm(rz_tilde - ((rz_tilde.T @ rx_t.T) * rx_t.T))
    ry = np.cross(rz.T,rx_t).T
    R_rect = np.vstack((rx_t,ry.T,rz.T))
    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2):
    # # TO DO
    sift = cv2.xfeatures2d.SIFT_create()
    kps = []
    h = np.shape(img1)[0]
    l = np.shape(img1)[1]
    X,Y = np.meshgrid(range(img1.shape[1]),range(img1.shape[0]),indexing='ij')
    XYZ_ind = np.array([X,Y]).T
    for i in range(np.shape(img1)[0]):
        for j in range(np.shape(img2)[1]):
            k = cv2.KeyPoint(XYZ_ind[i][j][0], XYZ_ind[i][j][1], _size=5)
            kps.append(k)
    kp1, des1 = sift.compute(img1, kps)
    kp2, des2 = sift.compute(img2, kps)
    des1_new = np.zeros((img1.shape[0], img1.shape[1], 128))
    des2_new = np.zeros((img2.shape[0], img2.shape[1], 128))
    for i in range(np.shape(img1)[0]):
        for j in range(np.shape(img1)[1]):
            des1_new[int(kp1[i*l + j].pt[1]), int(kp1[i*l + j].pt[0])] = des1[i*l + j]
            des2_new[int(kp2[i*l + j].pt[1]), int(kp2[i*l + j].pt[0])] = des2[i*l + j]
    disparity = np.zeros(np.shape(img1))
    for i in range(np.shape(img1)[0]):
        for j in range(np.shape(img1)[1]):
            t1 = np.reshape(des1_new[i,j], (1,128))
            t2 = des2_new[i,0:j+1]
            pixel_descriptor = np.linalg.norm((t2 - t1), axis=1)
            index = np.where(pixel_descriptor == pixel_descriptor.min())[0]
            disparity[i,j] = np.abs(index-j).min()
    return disparity

def points_colab():
    pts1 = np.array([[8, 482], [323, 680], [330, 632], [78, 477], [657, 461], [690, 439], [867, 321], [893, 300], [362, 272], [378, 258], [715, 51], [689, 63], [434, 209], [458, 197]])
    pts2 = np.array([[572, 81], [233, 271], [282, 271], [569, 120], [533, 518], [566, 548], [733, 679], [778, 713], [897, 322], [917, 339], [1270, 599], [1228, 568], [983, 389], [1013, 412]])
    
    pts1 = np.array([[82, 477], [411, 668], [163, 472], [415, 629], [745, 457], [787, 431], [428, 263], [463, 244], [595, 257], [951, 320], [987, 291], [530, 203], [553, 190], [772, 57], [790, 42], [940, 285], [581, 2], [615, 2], [590, 257], [584, 122], [609, 123], [464, 428], [505, 399], [588, 193], [602, 193], [576, 52], [614, 53]])
    pts2 = np.array([[428, 43], [75, 229], [426, 72], [139, 235], [381, 480], [422, 511], [754, 281], [785, 301], [505, 238], [597, 638], [634, 667], [837, 343], [879, 374], [1086, 529], [1127, 560], [651, 636], [515, 3], [483, 1], [500, 237], [515, 111], [492, 113], [482, 288], [511, 315], [513, 174], [496, 173], [519, 55], [483, 55]])
    return pts1, pts2


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    plt.figure(1)
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    plt.figure(2)
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    plt.figure(3)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    plt.figure(4)
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    plt.figure(5)
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.figure(6)
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    # img_left = cv2.imread('./left.jpg', 1)
    # img_right = cv2.imread('./right.jpg', 1)
    img_left = cv2.imread('./Figure_1.png', 1)
    img_right = cv2.imread('./Figure_2.png', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    # pts1, pts2 = find_match(img_left, img_right)
    pts1, pts2 = points_colab()
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[100*4.1012, 0, 1280/2], [0, 100*2.3069, 720/2], [0, 0, 1]])
    K = np.array([[0.3204, 0, 1280/2], [0, 0.3204, 720/2], [0, 0, 1]])
    K = np.array([[312, 0, 1920/2], [0, 312, 1080/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    # pts3Ds = []
    # P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # for i in range(len(Rs)):
    #     P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
    #     pts3D = triangulation(P1, P2, pts1, pts2)
    #     pts3Ds.append(pts3D)
    # visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # # Step 5: disambiguate camera poses
    # R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # # Step 6: rectification
    # H1, H2 = compute_rectification(K, R, C)
    # img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    # img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    # visualize_img_pair(img_left_w, img_right_w)

    # # Step 7: generate disparity map
    # img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    # img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    # img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    # img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    # disparity = dense_match(img_left_w, img_right_w)
    # visualize_disparity_map(disparity)

    # # save to mat
    # sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
    #                                   'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
