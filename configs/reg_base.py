CFG = dict(
    model_name="x3d_xs",

    # training phases
    freeze=True,
    epochs_head=8,
    epochs_ft=20,

    # optimization
    lr_head=1e-3,
    lr_backbone=3e-5,
    weight_decay=1e-2,

    # loss
    huber_beta=0.2,
    head_dropout=0.5,

    # early stop + scheduler
    early_patience=7,
    min_delta=1e-4,
    sched_patience=3,

    # video sampling / input
    clip_duration=2.6,   # irrelevant in uniform mode (kept for fallback)
    num_frames=24,       # number of frames across whole video
    image_size=182,

    clips_train=1,      
    clips_val=1,

    batch_size=1,
    num_workers=0,

    # LOW TEMPORAL FREQUENCY SETTINGS
    sampling_mode="uniform",        # "contiguous" or "uniform"
    uniform_eps_sec=1.0/30.0,       # small time window to grab a frame
    uniform_jitter=0.2,             # random within each temporal segment

    # logging
    seed=0,
    runs_dir="runs_reg",
    results_csv="results/results_reg.csv",
    print_every_epoch=True,
)
