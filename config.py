import os
from pathlib import Path

class Config:

    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
    
    # Data processing
    IMAGE_SIZE = (512, 512)
    CHANNELS = 1  
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Dataset configuration
    DATASET_NAME = "nih_chest_xray"
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    

    CLASS_LABELS = [
        'No Finding',
        'Atelectasis',
        'Cardiomegaly', 
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia'
    ]
    
    # Data imbalance handling to be implemented
    IMBALANCE_STRATEGIES = [
        'weighted_loss',
        'smote',
        'class_weights',
        'data_augmentation'
    ]
    
    # Segmentation models to be used - #TODO  still needs searching -
    BACKBONE_MODELS = [
        'resnet50',
        'efficientnet_b3',
        'densenet121',
        'vit_base_patch16_224'
    ]
    MEDSAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    MEDSAM_MODEL_TYPE = "vit_h"
    # Training parameters
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    WEIGHT_DECAY = 1e-5


    SEGMENTATION_THRESHOLD = 0.5
    MIN_MASK_AREA = 100
    
    CLIP_MODEL = "ViT-B/32"
    TEXT_PROMPTS = [
        "lung opacity",
        "enlarged heart",
        "pleural effusion", 
        "pneumonia",
        "normal lung",
        "chest abnormality"
    ]
    

    CLASSIFICATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc_roc',
        'auc_pr'
    ]
    
    SEGMENTATION_METRICS = [
        'dice_coefficient',
        'iou'
    ]
    

    DEVICE = "cuda" if os.environ.get('CUDA_AVAILABLE', 'false').lower() == 'true' else "cpu"
    MIXED_PRECISION = True
    

    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = ['.dcm', '.DCM', '.png', '.jpg', '.jpeg']
    
    @classmethod
    def create_directories(cls):
        """Create necessary project directories"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.EXPERIMENTS_DIR,
            cls.PROCESSED_DATA_DIR / "images",
            cls.PROCESSED_DATA_DIR / "annotations",
            cls.MODELS_DIR / "checkpoints",
            cls.MODELS_DIR / "pretrained"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print("Project directories created successfully!")
    

    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        assert cls.TRAIN_SPLIT + cls.VAL_SPLIT + cls.TEST_SPLIT == 1.0, \
            "Data splits must sum to 1.0"
        
        assert cls.IMAGE_SIZE[0] > 0 and cls.IMAGE_SIZE[1] > 0, \
            "Image size must be positive"
        
        assert cls.BATCH_SIZE > 0, "Batch size must be positive"
        
        assert 0 < cls.LEARNING_RATE < 1, "Learning rate must be between 0 and 1"
        
        print("Configuration validation passed!")


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    BATCH_SIZE = 4  # Smaller batch for development
    EPOCHS = 5
    EXPERIMENT_TRACKING = False


config = DevelopmentConfig()

if __name__ == "__main__":

    config.create_directories()
    config.validate_config()
    
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Using device: {config.DEVICE}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")