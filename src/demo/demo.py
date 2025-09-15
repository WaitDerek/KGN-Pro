import pdb

import os, sys
import os.path as osp
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
sys.path.append('../src/lib')
import json

import numpy as np

from mayavi import mlab
import cv2


from opts import opts
from keypoint_graspnet import KeypointGraspNet as KGN
from utils.ddd_utils import depth2pc

from physical.insp_results import load_results, get_paths 
from physical.utils_physical import quad2homog, draw_scene

from matplotlib.backends.backend_pdf import PdfPages

class OptDemo(opts):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("--demo_rgb_path", type=str, default=None)
        self.parser.add_argument("--demo_depth_path", type=str, default=None)
        self.parser.add_argument("--demo_cam_path", type=str, default=None)
        self.parser.add_argument("--demo_data_folder", type=str, default=None)
    
    def parse(self, args=''):
        opt = super().parse(args)
        # dirname = osp.dirname(opt.rgb_path)
        # basename_noExt = osp.splitext(os.path.basename(opt.rgb_path))[0]

        if opt.demo_data_folder is None:
            assert opt.demo_rgb_path is not None and opt.demo_depth_path is not None and opt.demo_cam_path is not None, \
                "Please provide either a folder containing all demo data or individual data rgb, depth, and camera info path"

        return opt


def prepare_kgn(opt):
    kgn = KGN.buildFromArgs(opt)
    return kgn


def main(opt):

    # prepare kgn
    kgn = prepare_kgn(opt)

    # get data paths
    if opt.demo_data_folder is not None:
        rgb_paths, dep_paths, cam_info_paths = get_paths(opt.demo_data_folder)
    else:
        rgb_paths, dep_paths, cam_info_paths = [opt.demo_rgb_file], [opt.demo_depth_file], [opt.demo_cam_file]

    # run
    # for rgb_path ,dep_path, poses_path in zip(rgb_paths, dep_paths, cam_info_paths):
    #     rgb, dep, intrinsic, _, _, _ = load_results(rgb_path,dep_path, poses_path)
    #     intrinsic_pro = np.array([[616.36529541,   0.        , 310.25881958],
    #                                     [  0.        , 616.20294189, 236.59980774],
    #                                     [  0.        ,   0.        ,   1.        ]])
    #     breakpoint()
    #     kgn.set_cam_intrinsic_mat(intrinsic_pro)
    for rgb_path ,dep_path, _ in zip(rgb_paths, dep_paths, cam_info_paths):
        rgb_path = "demo/1/color_img_0.png"
        dep_path = "demo/1/depth_raw_0.npy"
        rgb, dep = load_results(rgb_path, dep_path)
        intrinsic = np.array([[616.36529541, 0.0, 310.25881958], [0.0, 616.20294189, 236.59980774], [0.0, 0.0, 1.0]])
        kgn.set_cam_intrinsic_mat(intrinsic)
        # run KGN
        input = np.concatenate([rgb.astype(np.float32), dep[:, :, None]], axis=2)
        quaternions, locations, widths, kpts_2d_pred, scores, reproj_errors, centers_keep= kgn.generate(input, return_all=True)
        widths += 0.05  # increment the open width, as the labeled widths are too tight.
        grasp_poses_pred_cam = quad2homog(
            locations=locations, 
            quaternions=quaternions
        )

        # filter far-away grasps that is off the ROI
        trl_norm = np.linalg.norm(grasp_poses_pred_cam[:, :3, 3], axis=1)
        filtered_idx = trl_norm > 0.9
        grasp_poses_pred_cam = grasp_poses_pred_cam[~filtered_idx, :, :]
        widths = widths[~filtered_idx]

        # get the point cloud
        pcl_cam = depth2pc(dep, intrinsic, frame="camera", flatten=True)
        invalid_pcl_idx = np.all(pcl_cam == 0, axis=1)
        pcl_cam = pcl_cam[~invalid_pcl_idx, :]
        pc_color = rgb.reshape(-1, 3)[~invalid_pcl_idx, :]

        # visualize
        # draw_scene(pcl_cam, grasps=grasp_poses_pred_cam, pc_color=pc_color, widths=widths)  
        print("Close the window to see the next")
        # mlab.show()

        if rgb_path == "demo/1/color_img_0.png":
            image_path = os.path.join(opt.demo_data_folder, "color_img_{}.png").format(0) 
            image = cv2.imread(image_path)
            image_keypoint = cv2.imread(image_path)
            # file_path = "demo/scene_info.json"
            # with open(file_path, 'r') as f:
            #     scene_info = json.load(f)
            # pose_gt_0 = scene_info['grasp_poses'][0][0]
            # pose_gt_1 = scene_info['grasp_poses'][1][0]
            # pose_gt_2 = scene_info['grasp_poses'][2][0]
            # pose_gt_3 = scene_info['grasp_poses'][3][0]
            # pose_gt_4 = scene_info['grasp_poses'][4][0]
            # pose_gt_5 = scene_info['grasp_poses'][5][0]
            
            if image is None:
                print(f"Image not found at {image_path}")
                continue
            
            max_groups = min(len(kpts_2d_pred), 30)
            if len(kpts_2d_pred) > 0:
                processed_centers = []
                min_center_distance = 20  
                
                for i, kpt_group in enumerate(kpts_2d_pred):
                    if i >= max_groups:
                        break
                    current_center = (float(centers_keep[i][0][0]), 
                                    float(centers_keep[i][0][1]))
                    

                    # too_close = False
                    # for existing_center in processed_centers:
                    #     distance = np.sqrt((current_center[0] - existing_center[0])**2 +
                    #                     (current_center[1] - existing_center[1])**2)
                    #     print(distance)
                    #     if distance < min_center_distance:
                    #         too_close = True
                    #         print("too close")
                    #         break
                    # if too_close:
                    #     continue  
                    
                    processed_centers.append(current_center)

                    # if(len(processed_centers) > 6):
                    #     break
                    
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 204, 255)] 
                    color = colors[1]
                    if len(kpt_group) >= 4:
                        p1 = (int(kpt_group[0][0]), int(kpt_group[0][1]))  
                        p2 = (int(kpt_group[1][0]), int(kpt_group[1][1]))  
                        p3 = (int(kpt_group[2][0]), int(kpt_group[2][1]))  
                        p4 = (int(kpt_group[3][0]), int(kpt_group[3][1]))  

                        # 1 -> 2
                        cv2.line(image, p1, p3, color, 2)
                        # 2 -> 4
                        cv2.line(image, p3, p4, color, 2)
                        # 4 -> 3
                        cv2.line(image, p4, p2, color, 2)

                        mid_point = ((p3[0] + p4[0]) // 2, (p3[1] + p4[1]) // 2)

                        dir_vector = (p3[0] - p1[0], p3[1] - p1[1])

                        line_length = int(np.sqrt(dir_vector[0] ** 2 + dir_vector[1] ** 2))

                        start_point = mid_point
                        end_point = (
                            mid_point[0] + dir_vector[0],
                            mid_point[1] + dir_vector[1]
                        )

                        start_point = (int(start_point[0]), int(start_point[1]))
                        end_point = (int(end_point[0]), int(end_point[1]))

                        cv2.line(image, start_point, end_point, color, 2)

                        for point in kpt_group:
                            cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)

                        cv2.circle(image_keypoint, (int(centers_keep[i][0][0]), int(centers_keep[i][0][1])), 5, colors[3], -1)
                        p_center = (int(centers_keep[i][0][0]), int(centers_keep[i][0][1]))
                        cv2.line(image_keypoint, p_center, p1, colors[2], 2)
                        cv2.line(image_keypoint, p_center, p2, colors[2], 2)
                        cv2.line(image_keypoint, p_center, p3, colors[2], 2)
                        cv2.line(image_keypoint, p_center, p4, colors[2], 2)
                        for point in kpt_group:
                            cv2.circle(image_keypoint, (int(point[0]), int(point[1])), 4, color, -1)    

                        image_all = image.copy()
                        cv2.circle(image_all, (int(centers_keep[i][0][0]), int(centers_keep[i][0][1])), 5, colors[3], -1)
                        cv2.line(image_all, p_center, p1, colors[2], 2)
                        cv2.line(image_all, p_center, p2, colors[2], 2)
                        cv2.line(image_all, p_center, p3, colors[2], 2)
                        cv2.line(image_all, p_center, p4, colors[2], 2)
                        for point in kpt_group:
                            cv2.circle(image_all, (int(point[0]), int(point[1])), 4, color, -1)    

                        output_image_path = os.path.join(opt.demo_data_folder, "model_last_{}.png".format(0))
                        cv2.imwrite(output_image_path, image)
                        output_image_path_keypoint = os.path.join(opt.demo_data_folder, "keypoints.png")
                        cv2.imwrite(output_image_path_keypoint, image_keypoint)
                        output_image_path_keypoint = os.path.join(opt.demo_data_folder, "grasp.png")
                        cv2.imwrite(output_image_path_keypoint, image_all)



            break

            # cv2.imshow('Keypoints with Grasp', image)
            # while True:
            #     if cv2.getWindowProperty('Keypoints with Grasp', cv2.WND_PROP_VISIBLE) < 1:
            #         break
            #     cv2.waitKey(100)
            # cv2.destroyAllWindows()

    return


if __name__=="__main__":
    args = OptDemo().init()
    print(args)
    main(args)