class ParameterSetting_AdvancedBarking():
    def __init__(self, dvc_root, csv_root, name_prefix, save_root, exp_name, pretrained, epochs):

        self.dvc_root = dvc_root
        self.csv_root = csv_root
        self.name_prefix = name_prefix
        self.save_root = save_root
        self.exp_name = exp_name
        self.pretrained = pretrained
        self.epochs = epochs
        self.task = "AdvancedBarking"

        self.batch_size = 128
        self.lr = 0.0001
        self.num_class = 4
        self.category = ['barking', 'howling', 'crying', 'others']

        self.weight_loss = None
        self.optimizer = 'adam'
        self.scheduler = 'cosine'

        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2
        self.model_arch = 'cnn14'

        self.sr = 8000
        self.nfft = 200
        self.hop = 80
        self.mel = 64
        self.inp = 500
        self.normalize_num = 32768.0

        self.preload = True
        self.sampler = True
        self.warmup = False
        self.spec_aug = True

class ParameterSetting_GlassBreaking():
    def __init__(self, dvc_root, csv_root, name_prefix, save_root, exp_name, pretrained, epochs):

        self.dvc_root = dvc_root
        self.csv_root = csv_root
        self.name_prefix = name_prefix
        self.save_root = save_root
        self.exp_name = exp_name
        self.pretrained = pretrained
        self.epochs = epochs
        self.task = "GlassBreaking"

        self.batch_size = 128
        self.lr = 0.0001
        self.num_class = 2
        self.category = ['glassbreaking', 'others']

        self.weight_loss = None
        self.optimizer = 'adam'
        self.scheduler = 'cosine'

        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2
        self.model_arch = 'cnn14'

        self.sr = 8000
        self.nfft = 200
        self.hop = 80
        self.mel = 64
        self.inp = 500
        self.normalize_num = 32768.0

        self.preload = True
        self.sampler = True
        self.warmup = False
        self.spec_aug = True

class ParameterSetting_HomeEmergency():
    def __init__(self, dvc_root, csv_root, name_prefix, save_root, exp_name, pretrained, epochs):

        self.dvc_root = dvc_root
        self.csv_root = csv_root
        self.name_prefix = name_prefix
        self.save_root = save_root
        self.exp_name = exp_name
        self.pretrained = pretrained
        self.epochs = epochs
        self.task = "HomeEmergency"

        self.batch_size = 128
        self.lr = 0.0001
        self.num_class = 2
        self.category = ['CoSmoke', 'others']

        self.weight_loss = None
        self.optimizer = 'adam'
        self.scheduler = 'cosine'

        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2
        self.model_arch = 'vggish'

        self.sr = 8000
        self.nfft = 1024
        self.hop = 512
        self.mel = 128
        self.inp = 128
        self.normalize_num = 200000.0

        self.preload = True
        self.sampler = True
        self.warmup = False
        self.spec_aug = True

class ParameterSetting_HomeEmergency_JP():
    def __init__(self, dvc_root, csv_root, name_prefix, save_root, exp_name, pretrained, epochs):

        self.dvc_root = dvc_root
        self.csv_root = csv_root
        self.name_prefix = name_prefix
        self.save_root = save_root
        self.exp_name = exp_name
        self.pretrained = pretrained
        self.epochs = epochs
        self.task = "HomeEmergency_JP"

        self.batch_size = 128
        self.lr = 0.0001
        self.num_class = 2
        self.category = ['CoSmoke', 'others']

        self.weight_loss = None
        self.optimizer = 'adam'
        self.scheduler = 'cosine'

        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2
        self.model_arch = 'cnn14'

        self.sr = 8000
        self.nfft = 200
        self.hop = 80
        self.mel = 64
        self.inp = 500
        self.normalize_num = 32768.0

        self.preload = True
        self.sampler = True
        self.warmup = False
        self.spec_aug = True

class ParameterSetting_Integration():
    def __init__(self, dvc_root, csv_root, name_prefix, save_root, exp_name, pretrained, epochs):

        self.dvc_root = dvc_root
        self.csv_root = csv_root
        self.name_prefix = name_prefix
        self.save_root = save_root
        self.exp_name = exp_name
        self.pretrained = pretrained
        self.epochs = epochs
        self.task = "Integrate"

        self.batch_size = 128
        self.lr = 0.0001
        self.num_class = 3
        self.category = ['CoSmoke', 'GlassBreaking', 'others']

        self.weight_loss = None
        self.optimizer = 'adam'
        self.scheduler = 'cosine'

        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2
        self.model_arch = 'cnn14'

        self.sr = 8000
        self.nfft = 200
        self.hop = 80
        self.mel = 64
        self.inp = 500
        self.normalize_num = 32768.0

        self.preload = True
        self.sampler = True
        self.warmup = False
        self.spec_aug = True