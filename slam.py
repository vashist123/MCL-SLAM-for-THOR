import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging

cos, sin = np.cos, np.sin
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """

    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szy = int(np.ceil((s.ymax - s.ymin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.float64)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.free_prob_thresh = 0.4
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))
        s.log_odds_thresh_free = np.log(s.free_prob_thresh / (1 - s.free_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """


        x = np.clip(x, a_min=s.xmin, a_max=s.xmax)
        y = np.clip(y, a_min=s.ymin, a_max=s.ymax)

        grid_x = np.floor((x - s.xmin) / s.resolution)
        grid_y = np.floor((y - s.ymin) / s.resolution)

        grid_xy = np.vstack((grid_x, grid_y))

        return grid_xy.astype(np.int64)

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """

    def __init__(s, resolution=0.05, Q=1e-3 * np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)
        s.best_particle = None
        s.best_particle_xy = None

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d' % (split, split, idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d' % (split, split, idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t'] - t))
        s.init_flag = False

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93+0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135, 135 + s.lidar_angular_resolution,
                                   s.lidar_angular_resolution) * np.pi / 180.0
        s.new_lidar_angs = None

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1 / 9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n) / float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """

        idx_of_part = np.random.choice(a=range(len(w)), size=len(w), p=w)
        p_new = np.take(p, idx_of_part, axis=1)
        resampled_weights = np.ones(w.shape) / w.shape[0]

        return p_new, resampled_weights

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w - w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)

        # 1. from lidar distances to points in the LiDAR frame
        lidar_xyz = np.zeros((3, len(angles)))  # z=0 since it is a planar lidar
        lidar_xyz[0, :] = d * cos(angles)
        lidar_xyz[1, :] = d * sin(angles)
        lidar_homogeneous = make_homogeneous_coords_3d(lidar_xyz)

        # 2. from LiDAR frame to the body frame
        translation_vec = np.array([0, 0, s.lidar_height])
        R_body_lidar = euler_to_se3(0, head_angle, neck_angle, translation_vec)
        body_homogeneous = R_body_lidar @ lidar_homogeneous

        # 3. from body frame to world frame
        translation_vec2 = np.array([p[0], p[1], s.head_height])
        R_world_body = euler_to_se3(0, 0, p[2], translation_vec2)
        world_homogeneous = R_world_body @ body_homogeneous


        world_xy_valid = world_homogeneous[:, world_homogeneous[2] > 0.1]
        return world_xy_valid[:2]


    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)
        return smart_minus_2d(s.lidar[t]['xyth'], s.lidar[t - 1]['xyth'])

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """

        delta = s.get_control(t)
        for i in range(s.p.shape[1]):
            s.p[:, i] = smart_plus_2d(s.p[:, i], delta)
            s.p[:, i] = smart_plus_2d(s.p[:, i], np.random.multivariate_normal(np.zeros(3), s.Q))

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        log_w_k1_k1 = np.log(w) + obs_logp
        normalised_log_w_k1_k1 = log_w_k1_k1 - slam_t.log_sum_exp(log_w_k1_k1)
        return np.exp(normalised_log_w_k1_k1)


    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        joint_t_idx = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        head_angle_t = s.joint['head_angles'][1][joint_t_idx]
        neck_angle_t = s.joint['head_angles'][0][joint_t_idx]

        P_k1_k1 = np.zeros(s.p.shape[1])
        for i in range(s.p.shape[1]):
            projection = s.rays2world(p = s.p[:, i], d = s.lidar[t]['scan'], head_angle= head_angle_t, neck_angle= neck_angle_t, angles =s.lidar_angles)
            xy_idx = s.map.grid_cell_from_xy(projection[0, :], projection[1, :])

            if s.init_flag == False:
                particle = s.p[:,0]
                s.best_particle = s.get_grid(particle[0], particle[1])
                x_free = np.ndarray.flatten(np.linspace(s.best_particle[0], xy_idx[0,:], endpoint=False).astype('int64'))
                y_free = np.ndarray.flatten(np.linspace(s.best_particle[1], xy_idx[1,:], endpoint=False).astype('int64'))
                s.map.cells[:, :] = 0.5
                s.map.cells[x_free, y_free] = 0
                s.map.cells[xy_idx[0, :], xy_idx[1, :]] = 1
                break
            mappp = s.map.cells[xy_idx[0,:],xy_idx[1,:]]
            P_k1_k1[i] = len(mappp[mappp > 0.6])

        if s.init_flag is True:
            s.w = s.update_weights(s.w, P_k1_k1)
            best_idx = np.argmax(s.w)
            best_projection = s.rays2world(s.p[:, best_idx], s.lidar[t]['scan'],head_angle= head_angle_t, neck_angle= neck_angle_t, angles =s.lidar_angles)
            best_xy_idx = s.map.grid_cell_from_xy(best_projection[0, :], best_projection[1, :])

            particle = s.p[:, best_idx]
            s.best_particle = s.get_grid(particle[0], particle[1])

            x_free = np.ndarray.flatten(
                np.linspace(s.best_particle[0], best_xy_idx[0,:], endpoint=False).astype('int64'))
            y_free = np.ndarray.flatten(
                np.linspace(s.best_particle[1], best_xy_idx[1,:], endpoint=False).astype('int64'))

            s.map.log_odds[best_xy_idx[0, :], best_xy_idx[1, :]] += s.lidar_log_odds_occ
            s.map.log_odds[x_free, y_free] += 0.1*s.lidar_log_odds_free
            s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)

            # creating binary map using log odds map

            s.map.cells[:, :] = 0.5
            s.map.cells[np.where(s.map.log_odds >= s.map.log_odds_thresh)] = 1
            s.map.cells[np.where(s.map.log_odds < -s.map.log_odds_thresh)] = 0


            s.resample_particles()
        s.init_flag = True

    def get_grid(s, x, y):
        grid_cell_idx = np.zeros((2)).astype('int64')

        grid_cell_idx[0] = np.clip(((x - s.map.xmin)/40)*s.map.szx , 0, s.map.szx - 1)
        grid_cell_idx[1] = np.clip(((y - s.map.ymin)/40)*s.map.szy , 0, s.map.szy - 1)
        return grid_cell_idx

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1 / np.sum(s.w ** 2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e / s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
