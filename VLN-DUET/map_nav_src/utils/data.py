import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import csv
import base64
import random

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size, aug_ft_file=None, aug_prob = 0.5, aug_dataset=False, partial_aug=-1):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self.aug_ft_file = aug_ft_file
        self._feature_store_aug = []
        self.aug_prob = aug_prob
        self.aug_dataset = aug_dataset
        self.scans = set()

        if self.aug_ft_file is not None:
            for aug_file in self.aug_ft_file:
                aug_feature_tmp = dict()
                with open(aug_file, 'r') as f:
                    TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
                    reader = csv.DictReader(f, delimiter='\t', fieldnames=TSV_FIELDNAMES)
                    for item in reader:
                        scan = item["scanId"]
                        vp = item["viewpointId"]
                        key = '%s_%s' % (scan, vp)
                        aug_feature_tmp[key] = np.frombuffer(base64.b64decode(item['features']),
                                                         dtype=np.float32).reshape((36, self.image_feat_size))
                        self.scans.add(scan)
                self._feature_store_aug.append(aug_feature_tmp)

        self.multi = multi
        self.partial_aug = partial_aug
        if self.partial_aug != -1:
            self.aug_scans = random.sample(self.scans, partial_aug)
        else:
            self.aug_scans = self.scans

    def get_image_feature(self, scan, viewpoint, ori=False, aug=False):
        key = '%s_%s' % (scan, viewpoint)
        if self.multi:
            if key in self._feature_store:
                ft = self._feature_store[key]
            else:
                with h5py.File(self.img_ft_file, 'r') as f:
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    self._feature_store[key] = ft
            index = random.randrange(len(self._feature_store_aug))
            ft_aug = self._feature_store_aug[index][key]
            ft = np.concatenate([ft, ft_aug], axis=-1)
        else:
            if key in self._feature_store:
                ft = self._feature_store[key]
            else:
                with h5py.File(self.img_ft_file, 'r') as f:
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    self._feature_store[key] = ft
            if scan in self.aug_scans:
                if not ori and self.aug_ft_file is not None and random.random() < self.aug_prob:
                    index = random.randrange(len(self._feature_store_aug))
                    ft = self._feature_store_aug[index][key]

                if aug or self.aug_dataset:
                    index = random.randrange(len(self._feature_store_aug))
                    ft = self._feature_store_aug[index][key]

        return ft


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

Matsim_version = 2
if Matsim_version == 1:
    import sys
    sys.path.append("~/R2R-EnvDrop/build")

import MatterSim

def new_simulator(connectivity_dir, scan_data_dir=None):
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    if Matsim_version == 1:
        sim.init()
    else:
        sim.setBatchSize(1)
        sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            if Matsim_version == 1:
                sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
            else:
                sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            if Matsim_version == 1:
                sim.makeAction(0, 1.0, 1.0)
            else:
                sim.makeAction([0], [1.0], [1.0])
        else:
            if Matsim_version == 1:
                sim.makeAction(0, 1.0, 0)
            else:
                sim.makeAction([0], [1.0], [0])

        if Matsim_version == 1:
            state = sim.getState()
        else:
            state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

