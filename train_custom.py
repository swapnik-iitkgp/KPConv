import time
import os

# Import configuration and trainer utilities from KPConv
from utils.config import Config
from utils.trainer import ModelTrainer
from models.KPFCNN_model import KernelPointFCNN

# Import our custom dataset class
from datasets.CustomS3DISDataset import CustomS3DISDataset

class CustomS3DISConfig(Config):
    """
    Configuration for training on the custom S3DIS-format data.
    Number of classes is set to 4 (labels 0,1,2,3).
    """
    ####################
    # Dataset parameters
    ####################
    dataset = 'CustomS3DIS'
    num_classes = 4  # Our data has 4 labels: 0,1,2,3
    network_model = 'cloud_segmentation'
    input_threads = 4

    #########################
    # Architecture definition
    #########################
    architecture = [
        'simple',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary'
    ]
    num_kernel_points = 15
    first_subsampling_dl = 0.04
    in_radius = 2.0
    density_parameter = 5.0
    KP_influence = 'linear'
    KP_extent = 1.0
    convolution_mode = 'sum'
    modulated = False
    offsets_loss = 'fitting'
    offsets_decay = 0.1
    in_features_dim = 5  # You may adjust according to your input features
    use_batch_norm = True
    batch_norm_momentum = 0.98

    #####################
    # Training parameters
    #####################
    max_epoch = 500
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1/100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0
    batch_num = 10
    epoch_steps = 500
    validation_size = 50
    snapshot_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8

    batch_averaged_loss = False
    saving = True
    saving_path = None


if __name__ == '__main__':

    ##########################
    # Initiate environment
    ##########################
    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load configuration
    ###########################
    config = CustomS3DISConfig()

    ##############
    # Prepare Dataset
    ##############
    print('\nDataset Preparation')
    print('*******************')
    # Specify the root folder of your custom S3DIS-format data.
    # For example, the folder structure should contain subfolders like "Area_1" with room_*.txt files.
    dataset_root = r"/kaggle/working/S3DIS_format"
    dataset = CustomS3DISDataset(dataset_root, config.input_threads)
    dataset.load_rooms()           # Load all room files from the S3DIS-format structure
    dataset.init_input_pipeline(config)  # Prepare flat_inputs for training

    ##############
    # Define Model
    ##############
    print('\nCreating Model')
    print('**************\n')
    t1 = time.time()
    model = KernelPointFCNN(dataset.flat_inputs, config)
    trainer = ModelTrainer(model)
    t2 = time.time()
    print('\nModel created in {:.1f} s'.format(t2 - t1))
    
    ################
    # Start Training
    ################
    print('\nStart Training')
    print('**************\n')
    trainer.train(model, dataset)