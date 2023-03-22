import os
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import cv2
import copy, math
def writeply(filename,xyz,rgb):
    """write into a ply file"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


def get3d_box_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h


def get_obj_faces(object_center, w, l, h):
    # calculate all the vectrics
    z_1 = object_center[2] - h/2
    z_2 = object_center[2] + h/2
    x_1 = object_center[1] - l/2
    x_2 = object_center[1] + l/2
    y_1 = object_center[0] - w/2
    y_2 = object_center[0] + w/2
    
    # eight points
    p_1 = np.array([x_1,y_1,z_1])
    p_2 = np.array([x_1,y_1,z_2])
    p_3 = np.array([x_1,y_2,z_1])
    p_4 = np.array([x_1,y_2,z_2])
    p_5 = np.array([x_2,y_1,z_1])
    p_6 = np.array([x_2,y_1,z_2])
    p_7 = np.array([x_2,y_2,z_1])
    p_8 = np.array([x_2,y_2,z_2])

    # six face
    fx_1 = np.array([p_5, p_6, p_7, p_8])
    fx_2 = np.array([p_1, p_2, p_3, p_4])
    fy_1 = np.array([p_1, p_2, p_5, p_6])
    fy_2 = np.array([p_3, p_4, p_7, p_8])
    fz_1 = np.array([p_1, p_3, p_7, p_5])
    fz_2 = np.array([p_2, p_4, p_8, p_6])

    obj_faces = []
    obj_faces.append(fx_1)
    obj_faces.append(fx_2)
    obj_faces.append(fy_1)
    obj_faces.append(fy_2)
    obj_faces.append(fz_1)
    obj_faces.append(fz_2)
    return obj_faces

def calculateDistance(p1, p2):
    """
    Calculate distance between two points in the space/3D.
    """
    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return dist

def get_plane_area(plane):
    for i in range(3):  # loop on x, y, z
        if len(np.unique(plane[:, i])) == 1:
            break
    plane = np.delete(plane, i, axis=1)
    x1, x2 = plane[:, 0].min(), plane[:, 0].max()
    y1, y2 = plane[:, 1].min(), plane[:, 1].max()
    area = (y2-y1)*(x2-x1)
    return area

def get_plane_center(plane):
    # Get center of the 4 points/polygon:
    x = [p[0] for p in plane]
    y = [p[1] for p in plane]
    z = [p[2] for p in plane]
    centroid = np.array([sum(x) / len(plane), sum(y) / len(plane), sum(z) / len(plane)])
    return centroid

def get_nearst_face_from_point(point, faces, z=False):
    """
    Calculate distances between point in 3D and each face to choose the nearst face/plane.
    """
    if z == False:
        # Get the largest two faces:
        if get_plane_area(faces[0]) >= get_plane_area(faces[2]):
            faces = faces[:2]
        else:
            faces = faces[2:]
    min_dist = calculateDistance(point, get_plane_center(faces[0]))
    nearst_face = faces[0]
    for face in faces:
        dist = calculateDistance(point, get_plane_center(face))
        if dist < min_dist:
            min_dist = dist
            nearst_face = face
    return nearst_face


def get_angle_between_2vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * (180 / np.pi)

    return angle

def get_perpendicular_vector_on_plane(plane, point):
    """
    Get perpendicular vector on plane. Use the point to determine the direction of the vector.
    The plane is represented by 4 corners [4, 3].
    """
    O = get_plane_center(plane)
    #O = np.array([plane[0, 0], plane[0, 1], plane[0, 2]])  # Corner to be used as the origin
    V1 = np.array([plane[1, 0], plane[1, 1], plane[2, 2]]) - O  # Relative vectors
    V2 = np.array([plane[2, 0], plane[2, 1], plane[2, 2]]) - O
    V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
    V2 = V2 / np.linalg.norm(V2)
    # Take the cross product
    perp = np.cross(V1, V2)

    direction = perp / np.linalg.norm(perp)
    # To avoid looking from outside the room
    check_dir = [np.sign(point[0]-O[0]), np.sign(point[1]-O[1]), np.sign(point[2]-O[2])]
    for i in range(3):  # loop on x, y ,z
        if np.sign(direction[i]) != np.sign(check_dir[i]):
            direction[i] = direction[i] * -1
    return direction

def lookat(center, target, up):
    """
    https://github.com/isl-org/Open3D/issues/2338
    https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    https://www.youtube.com/watch?v=G6skrOtJtbM
    f: forward
    s: right
    u: up
    """
    f = (target - center)
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = -s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    t = np.matmul(-m[:3, :3], center)
    m[:3, 3] = t

    return m

def get_point_on_circle(point, circelCenter, angle):
    """
    get a point on the circle surface given the angle.
    https://stackoverflow.com/questions/58501322/how-to-calculate-point-on-circle-from-angle-between-middle-and-other-point-on-th
    """
    x1 = circelCenter[0] + (point[0] - circelCenter[0]) * math.cos(angle) - (point[1] - circelCenter[1]) * math.sin(angle)
    y1 = circelCenter[1] + (point[0] - circelCenter[0]) * math.sin(angle) + (point[1] - circelCenter[1]) * math.cos(angle)
    return [x1, y1]

def convert_world2image_cord_vectorized_ver(obj_pc_voxel, m, intrinsic):
    """
    This function exactly like "convert_world2image_cord" but this is the vectorized version of it.
    """
    projected_points = np.zeros_like(obj_pc_voxel)
    m = m[:3, :]
    m = np.repeat(m[np.newaxis, :, :], len(obj_pc_voxel), axis=0)  # [num_objs, 3, 4]
    intrinsic = np.repeat(intrinsic[np.newaxis, :, :], len(obj_pc_voxel), axis=0)  # [num_objs, 3, 3]
    obj_pc_voxel = np.hstack((obj_pc_voxel, np.ones((len(obj_pc_voxel), 1))))  # [num_objs, 4]
    p_cam = np.matmul(m, np.expand_dims(obj_pc_voxel, axis=-1))  # [num_objs, 3, 4].[num_objs, 4, 1]=[num_objs, 3, 1]
    p_img = np.matmul(intrinsic, p_cam)  # [num_objs, 3, 3].[num_objs, 3, 1] = [num_objs, 3, 1]
    p_pixel = p_img[:, :, 0] * (1 / p_img[:, -1, :])
    return p_pixel

def project_pc_2_img(scan, obj, saving_pth, count, save, augment=True, cocoonAngles = [0], ):

    scan_pc = scan[:, :3]
    scan_color = scan[:, 3:]
    # configurations:
    voxel_ratio_org = [2/100, 2/100, 2/100]
    k_size = 5
    desired_shape = (128,128)
    max_grid_dim = 1200
    up_vector = np.array([0, 0, -1])

    # get scene dimensions (w, l, h):
    w, l, h = get3d_box_from_pcs(scan_pc)
    obj_w, obj_l, obj_h = get3d_box_from_pcs(obj)
    # get center of the scene
    scene_center = np.array([scan_pc[:, 0].max() - w / 2, scan_pc[:, 1].max() - l / 2, scan_pc[:, 2].max() - h / 2])
    object_center = np.array([obj[:, 0].max() - obj_w / 2, obj[:, 1].max() - obj_l / 2, obj[:, 2].max() - obj_h / 2])
    # Get camera pos & the target point:
    # ----------------------------------
    faces = get_obj_faces(object_center, obj_w, obj_l, obj_h)[:4]  # exclude z faces  ##what do you get here?? 

    nearst_face = get_nearst_face_from_point(scene_center, faces)
    direction = get_perpendicular_vector_on_plane(plane=nearst_face, point=scene_center)
    box_center = object_center
    O = box_center
    
    intrinsic = np.array([[623.53829072, 0., 359.5], [0., 623.53829072, 359.5], [0., 0., 1.]])
    # Voxelizing the obj point-clouds:

    obj_pc = obj[:,:3]
    obj_color = obj[:,3:]
    voxel_ratio = copy.deepcopy(voxel_ratio_org)

    w, l, h = get3d_box_from_pcs(obj_pc)
    x_bound = [obj_pc[:, 0].min() - (w * voxel_ratio[0]), obj_pc[:, 0].max() + (w * voxel_ratio[0])]
    y_bound = [obj_pc[:, 1].min() - (l * voxel_ratio[1]), obj_pc[:, 1].max() + (l * voxel_ratio[1])]
    z_bound = [obj_pc[:, 2].min() - (h * voxel_ratio[2]), obj_pc[:, 2].max() + (h * voxel_ratio[2])]
    # filter the voxel from the whole scene:
    filtered_idx = np.where((scan_pc[:, 0] < x_bound[1]) & (scan_pc[:, 0] > x_bound[0])
                            & (scan_pc[:, 1] < y_bound[1]) & (scan_pc[:, 1] > y_bound[0])
                            & (scan_pc[:, 2] < z_bound[1]) & (scan_pc[:, 2] > z_bound[0]))
    obj_pc_voxel = scan_pc[filtered_idx]
    obj_color_voxel = scan_color[filtered_idx]

    # set the camera away from the object at certain distance (d)
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    if augment:
        d = np.random.uniform(1.5, 4)
        up_d = np.random.uniform(0.5, 2.5)
        dir_x = np.random.uniform(0.01, 0.2)
        dir_y = np.random.uniform(0.01, 0.2)
        direction[0] += dir_x
        direction[1] += dir_y
    else:
        d = 2
        up_d = 1
    camera_pos = O + (d * direction)

    # Take cocoon shots for the object: (Photo session :D)
    org_camera_pos = copy.deepcopy(camera_pos)
    for angle in cocoonAngles:
        if augment:
            added_angle = np.random.uniform(5, 25)
            added_angle += angle
        else:
            added_angle = angle
        camera_pos[:2] = get_point_on_circle(org_camera_pos[:2], O[:2], angle=added_angle * np.pi / 180)
        camera_pos[-1] = org_camera_pos[-1] + up_d  # lift the camera

        m = lookat(camera_pos, O, up_vector)

        projected_points = convert_world2image_cord_vectorized_ver(obj_pc_voxel, m, intrinsic)
        camProjected_points = copy.deepcopy(projected_points)

        # Shift -ve points:
        projected_points[:, 0] = projected_points[:, 0] - projected_points[:, 0].min()
        projected_points[:, 1] = projected_points[:, 1] - projected_points[:, 1].min()

        ptXYZRGB = np.hstack((projected_points, obj_color_voxel))
        ptXYZRGB_copy = copy.deepcopy(ptXYZRGB)

        grid = np.ones((min(math.ceil(ptXYZRGB[:, 1].max()) + k_size, max_grid_dim + k_size),
                        min(math.ceil(ptXYZRGB[:, 0].max()) + k_size, max_grid_dim + k_size), 3)) * 255
        # check grid boundaries:
        if math.ceil(ptXYZRGB[:, 1].max()) > max_grid_dim:
            ptXYZRGB[:, 1] = (ptXYZRGB[:, 1] / ptXYZRGB[:, 1].max()) * max_grid_dim
        if math.ceil(ptXYZRGB[:, 0].max()) > max_grid_dim:
            ptXYZRGB[:, 0] = (ptXYZRGB[:, 0] / ptXYZRGB[:, 0].max()) * max_grid_dim

        # Overlap the original object over the rest of the scene:
        projected_obj_points = convert_world2image_cord_vectorized_ver(obj_pc, m, intrinsic)
        projected_obj_points[:, 0] = projected_obj_points[:, 0] - camProjected_points[:, 0].min()
        projected_obj_points[:, 1] = projected_obj_points[:, 1] - camProjected_points[:, 1].min()
        objptXYZRGB = np.hstack((projected_obj_points, obj_color))
        if math.ceil(ptXYZRGB_copy[:, 1].max()) > max_grid_dim:
            objptXYZRGB[:, 1] = (objptXYZRGB[:, 1] / ptXYZRGB_copy[:, 1].max()) * max_grid_dim
        if math.ceil(ptXYZRGB_copy[:, 0].max()) > max_grid_dim:
            objptXYZRGB[:, 0] = (objptXYZRGB[:, 0] / ptXYZRGB_copy[:, 0].max()) * max_grid_dim

        x = -objptXYZRGB[:, 0]
        y = -objptXYZRGB[:, 1]
        r = objptXYZRGB[:, 3]
        g = objptXYZRGB[:, 4]
        b = objptXYZRGB[:, 5]

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=np.vstack((r, g, b)).T / 255.0, s=20)
        ax.set_axis_off()
        if save:
            fig.savefig(saving_pth + '/proposal_{}'.format(count) + "_" + 'angle_{}.jpg'.format(angle))
        
        plt.close()
       

def mask2images(mask_proposal, full_scans, Angles, Resize, background, save_image, name):
    '''
    This is a function that can conver mask proposal to images
    input:  mask_proposal : [N, n_proposal]
            full_scans : [N_large]
            Angles: [0, 30, -30]
            Resize: 1.5
            Background: True/False
            save_image: True/False

    output: image_level_features: [n_proposal, n_angles, CLIP features]
            global_level_features: [n_proposal, CLIP features afterpooling]
    '''
    for count, point in  enumerate(mask_proposal):
        # get the box size
        w, l, h = get3d_box_from_pcs(point)
        # get the center point
        object_center = np.array([point[:, 0].max() - w / 2, point[:, 1].max() - l / 2, point[:, 2].max() - h / 2])
        w = w * Resize
        l = l * Resize
        h= h * Resize
        #select points within double size of the box in the whole points
        index_pos = np.where((full_scans[:,0]>object_center[0]-w/2) & (full_scans[:,0]<object_center[0]+w/2) & (full_scans[:,1]>object_center[1]-l/2) & (full_scans[:,1]<object_center[1]+l/2) & (full_scans[:,2]>object_center[2]-h/2) & (full_scans[:,2]<object_center[2]+h/2))
        pc_w_bg = full_scans[index_pos]

        filename = f"project_2d_without_bg/{name}/"
        if not os.path.exists(filename):
            os.makedirs(filename)
        if background:
            project_pc_2_img(full_scans, pc_w_bg, filename, count, save_image, cocoonAngles = Angles)
        else: 
            project_pc_2_img(full_scans, point, filename, count, save_image, cocoonAngles = Angles)
