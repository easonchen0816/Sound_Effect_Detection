class ParameterSetting():
    def __init__(self, csv_root='./', name_prefix='furbo_only', save_root='snapshots',
                 epochs=20, batch_size=128, lr=0.0001, num_class=2,
                 weight_loss=1, time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2,
                 sr=8000, nfft=200, hop=80, mel=64, resume=None, pretrained=None, normalize=None, preload=False, spec_in_model=False,
                 sampler=False, warmup=False, spec_aug=False, mixup=False, optimizer='adam', scheduler='cosine', train_last_layer=False, h5=False, h5_path=None):

        self.csv_root = csv_root
        self.name_prefix = name_prefix
        self.save_root = save_root

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_class = num_class

        self.weight_loss = weight_loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

        self.sr = sr
        self.nfft = nfft
        self.hop = hop
        self.mel = mel

        self.resume = resume
        self.pretrained = pretrained
        self.normalize = normalize
        self.preload = preload
        self.spec_in_model = spec_in_model
        self.sampler = sampler
        self.warmup = warmup
        self.spec_aug = spec_aug
        self.mixup = mixup
        self.train_last_layer = train_last_layer
        self.h5 = h5
        self.h5_path = h5_path
