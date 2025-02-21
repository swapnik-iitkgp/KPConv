import os
import glob
import numpy as np
from plyfile import PlyData
from datasets.common import Dataset  # Use the common dataset base if available; otherwise, create a minimal one.

class CustomS3DISDataset(Dataset):
    """
    Custom dataset class for S3DIS-format data.
    Expected folder structure:
        root_dir/
            Area_1/
                room_*.txt
            Area_2/
                room_*.txt
            ...
    Each room file is expected to have lines:
        x y z r g b label instance_id
    """
    def __init__(self, root_dir, input_threads=4):
        Dataset.__init__(self, 'CustomS3DIS')
        self.root_dir = root_dir
        self.input_threads = input_threads
        self.area_names = []   # Will be populated in load_rooms()
        self.room_files = []   # List of room file paths
        self.rooms = []        # Loaded room data
        self.flat_inputs = None  # To be prepared for training

        # Define label mapping for 4 classes
        self.label_to_names = {0: 'class0', 1: 'class1', 2: 'class2', 3: 'class3'}
        self.num_classes = 4

    def load_rooms(self):
        """
        Load all room files from the S3DIS-format structure.
        """
        # Look for all areas (folders starting with "Area_")
        area_folders = glob.glob(os.path.join(self.root_dir, "Area_*"))
        for area in area_folders:
            self.area_names.append(os.path.basename(area))
            room_files = glob.glob(os.path.join(area, "room_*.txt"))
            self.room_files.extend(room_files)
        print(f"Found {len(self.room_files)} room files in areas: {self.area_names}")

        # Load each room file into a list.
        self.rooms = []
        for room_file in self.room_files:
            try:
                data = np.loadtxt(room_file)
                # Expecting shape (N, 8): x, y, z, r, g, b, label, instance_id
                self.rooms.append(data)
            except Exception as e:
                print(f"Error loading {room_file}: {e}")
        print(f"Loaded {len(self.rooms)} rooms.")

    def init_input_pipeline(self, config):
        """
        For simplicity, concatenate all room data into one flat set.
        """
        all_points = []
        all_colors = []
        all_labels = []
        for room in self.rooms:
            all_points.append(room[:, :3])
            all_colors.append(room[:, 3:6])
            all_labels.append(room[:, 6])  # semantic labels
        self.all_points = np.concatenate(all_points, axis=0)
        self.all_colors = np.concatenate(all_colors, axis=0)
        self.all_labels = np.concatenate(all_labels, axis=0).astype(np.int32)
        print(f"Total points in dataset: {self.all_points.shape[0]}")

        # Prepare a simple flat input structure.
        self.flat_inputs = {
            'points': self.all_points,
            'colors': self.all_colors,
            'labels': self.all_labels
        }