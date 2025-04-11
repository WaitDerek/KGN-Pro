import pdb

import os, sys
import os.path as osp
src_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_path)
sys.path.append('../src/lib')

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

    # Initialize PdfPages to save the images
    with PdfPages("output_grasps.pdf") as pdf:
        
        for rgb_path, dep_path, _ in zip(rgb_paths, dep_paths, cam_info_paths):
            rgb, dep = load_results(rgb_path, dep_path)
            intrinsic = np.array([[616.36529541, 0.0, 310.25881958], [0.0, 616.20294189, 236.59980774], [0.0, 0.0, 1.0]])
            kgn.set_cam_intrinsic_mat(intrinsic)

            # run KGN
            input = np.concatenate([rgb.astype(np.float32), dep[:, :, None]], axis=2)
            quaternions, locations, widths, kpts_2d_pred, scores, output = kgn.generate(input)
            widths += 0.05  # increment the open width, as the labeled widths are too tight.
            grasp_poses_pred_cam = quad2homog(locations=locations, quaternions=quaternions)

            # filter far-away grasps that are off the ROI
            trl_norm = np.linalg.norm(grasp_poses_pred_cam[:, :3, 3], axis=1)
            filtered_idx = trl_norm > 0.9
            grasp_poses_pred_cam = grasp_poses_pred_cam[~filtered_idx, :, :]
            widths = widths[~filtered_idx]

            # get the point cloud
            pcl_cam = depth2pc(dep, intrinsic, frame="camera", flatten=True)
            invalid_pcl_idx = np.all(pcl_cam == 0, axis=1)
            pcl_cam = pcl_cam[~invalid_pcl_idx, :]
            pc_color = rgb.reshape(-1, 3)[~invalid_pcl_idx, :]

            # Read image
            if rgb_path == "demo/color_image_0.png":
                image_path = os.path.join(opt.demo_data_folder, "color_image_{}.png").format(0) 
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Image not found at {image_path}")
                    continue

                # Create a copy for the second image (with keypoints only)
                image_with_keypoints = image.copy()

                max_groups = min(len(kpts_2d_pred), 40)
                if len(kpts_2d_pred) > 0:
                    for i, kpt_group in enumerate(kpts_2d_pred):
                        if i >= 1:  # max
                            break

                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
                        color = colors[1]

                        if len(kpt_group) >= 4:
                            p1 = (int(kpt_group[0][0]), int(kpt_group[0][1]))  # 1
                            p2 = (int(kpt_group[1][0]), int(kpt_group[1][1]))  # 2
                            p3 = (int(kpt_group[2][0]), int(kpt_group[2][1]))  # 3
                            p4 = (int(kpt_group[3][0]), int(kpt_group[3][1]))  # 4
                            # 1 -> 2
                            cv2.line(image, p1, p3, color, 3)
                            # 2 -> 4
                            cv2.line(image, p3, p4, color, 3)
                            # 4 -> 3
                            cv2.line(image, p4, p2, color, 3)

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
                            cv2.line(image, start_point, end_point, color, 3)

                            for point in kpt_group:
                                cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)

                # Save the image with grasp lines
                pdf.savefig(image)  # Save this image to the PDF

                # 绘制仅含关键点的图像
                if len(kpts_2d_pred) > 0:
                    for i, kpt_group in enumerate(kpts_2d_pred):
                        if i >= 1:  # 限制最大组数
                            break
                        
                        # Draw only the four keypoints
                        if len(kpt_group) >= 4:
                            for point in kpt_group:
                                cv2.circle(image_with_keypoints, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

                # Save the image with keypoints only
                pdf.savefig(image_with_keypoints)  # Save this image to the PDF

    return



if __name__=="__main__":
    args = OptDemo().init()
    print(args)
    main(args)