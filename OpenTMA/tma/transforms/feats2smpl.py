from os.path import join as pjoin

import numpy as np
import torch

import tma.data.humanml.utils.paramUtil as paramUtil
from tma.data.humanml.data.dataset import Text2MotionDatasetV2
from tma.data.humanml.scripts.motion_process import recover_from_ric
from tma.data.humanml.utils.plot_script import plot_3d_motion

# Define the skeleton structure using the kinematic chain from paramUtil
skeleton = paramUtil.t2m_kinematic_chain


def main():
    # Define paths and parameters
    data_root = '../datasets/humanml3d'
    feastures_path = 'in.npy'
    animation_save_path = 'in.mp4'

    fps = 20
    # Load the mean and standard deviation of the dataset
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))

    # Load the motion features and normalize them using the mean and standard deviation
    motion = np.load(feastures_path)
    motion = motion * std + mean
    motion_rec = recover_from_ric(torch.tensor(motion), 22).cpu().numpy()

    # Scale the recovered motion
    motion_rec = motion_rec * 1.3
    # Plot and save the 3D motion
    plot_3d_motion(animation_save_path, motion_rec, title='input', fps=fps)


# Run the main function if the script is run as the main program
if __name__ == '__main__':
    main()
