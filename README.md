Polarized_Deblur/
│
├── data/                   # Dataset directory
│   ├── train/
│   ├── test/
│   └── dataloader.py       # Data loader
│
├── models/                 # Model definitions
│   ├── unpolarized_estimator.py  #  Unpolarized image estimator
│   ├── polarized_reconstructor.py  # Polarized image reconstructor
│   ├── saved_models/       # pretrained model
│   └── losses.py           # Loss functions definition
│
├── utils/                  # Utility functions
│   ├── visualization.py    # Visualization tools
│   └── metrics.py          # Evaluation metrics (e.g., PSNR, SSIM)
│
├── x_north_calculate/           # angle between the carrier and geographic north
│   ├── sun_north_calculate.m    # angle between the sun and geographic north
│   └── x_sun_calculate.m        # angle between the carrier and the sun
│
├── train.py                # Training script
├── test.py                 # Testing script
├── config.py               # Configuration file (hyperparameters)
└── README.md               # Project description
