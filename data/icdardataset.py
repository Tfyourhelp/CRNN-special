from lib import *
from utils.pre_processing.img_proc import processing_img
# from text_augmentation_main.warp import Curve, Distort, Stretch
# from text_augmentation_main.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
# from text_augmentation_main.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
# from text_augmentation_main.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
# from text_augmentation_main.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
# from text_augmentation_main.camera import Contrast, Brightness, JpegCompression, Pixelate
# from text_augmentation_main.weather import Fog, Snow, Frost, Rain, Shadow
# from text_augmentation_main.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color



class Icdar15Dataset(Dataset):

    CHARS = string.digits +  string.ascii_lowercase + string.ascii_uppercase + "-./:"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)} # char sang index
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()} # index sang char

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100, aug= False):

        # self.aug = aug # có aug hay không 


        # self.ops1 = [AutoContrast(), Invert(), Rotate(),ImpulseNoise()] # phép tạo ra 1 ảnh mới 
        # self.ops2 = ([Curve(), Perspective(), Distort(), Stretch(), TranslateX(), TranslateY(), Grid()]) # phép tạo ra 2 ảnh mới 
        # self.ops2.extend([GaussianBlur(), DefocusBlur(), MotionBlur()])#, GlassBlur() ,ZoomBlur()]
        # self.ops2.extend([Contrast(), Brightness(), JpegCompression(), Pixelate()])
        # self.ops2.extend([Fog(), Snow(), Frost(),Solarize()])

        if root_dir and mode and not paths: 
            paths, texts = self._load_from_raw_files(root_dir, mode)

#================ base ( base_data)
            if mode == 'train_aug': # train 250 ảnh tập special character 11 -->nhân  tính thêm aug
                paths = paths[2000:]
                texts = texts[2000:]
            
            elif mode == 'val_aug':  # valid 50 ảnh tập special character 11 -->nhân  tính thêm aug
                paths = paths[:500]
                texts = texts[:500]
            elif mode == 'test':  # đánh giá thì 10 ảnh kế cuối
                paths = paths[500:600]
                texts = texts[500:600]
            elif mode == 'train_anotherexp': #  train another thì từ 20 ảnh đầu đến hết 
                paths = paths[20:]
                texts = texts[20:]
            elif mode == 'val_anotherexp':
                paths = paths[0:20]
                texts = texts[0:20]

#================ fintunning icdar
            elif mode == 'train_icdar_special': # train full tập train icdar
                paths = paths[:]
                texts = texts[:]
            elif mode == 'val_icdar_special': # valid full tập train icdar
                paths = paths[:]
                texts = texts[:]



        elif not root_dir and not mode and paths: 
            texts = None
            
        #print(texts)
        self.paths = paths
        self.texts = texts
        # print(self.paths)
        # print(self.texts)
        self.img_height = img_height
        self.img_width = img_width




    def _load_from_raw_files(self, root_dir, mode):

        paths_file = None
        if mode == 'train_aug' or mode == 'val_aug' or mode == 'train_anotherexp' or mode == 'val_anotherexp' :
            paths_file = 'gt.txt'
      
        elif mode == 'train_icdar_special':
            paths_file = 'gt_train.txt' # text-reg-data/icdar/gt_train.txt
        elif mode == 'val_icdar_special':
            paths_file = 'gt_valid.txt'  # text-reg-data/icdar/gt_val.txt



        print(root_dir)
        with open(os.path.join(root_dir, paths_file), 'r', encoding='utf-8-sig') as f:
            
            paths_texts = [line.strip().split(', ') for line in f]
            #print(paths_texts[:70])
            paths = []
            texts = []
            
            
            for i in paths_texts:
                t = i[1].replace('"', '')
                if all(c in self.CHARS for c in t): # bỏ qua những path có kí tự đặc biệt 
                    paths.append(os.path.join(root_dir, i[0]))
                    texts.append(t)


        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        print(path)
        try:
            image = Image.open(path).convert('L')  # grey-scale
    
        except IOError:
            print(path)
            print('Corrupted image for %d' % index)
            return self[index + 1]
        
        image = np.array(image) # bởi vì đọc ảnh = PIL thì không phải numpy , nên là muốn nó có .shape thì numpy array
       
        #image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = processing_img(image , self.img_width, self.img_height, 0)
        #print(image.shape)
        
        image = image.reshape((1, self.img_height, self.img_width)) #?? chỗ này là sao
        image = (image / 127.5) - 1.0 #normalize --> co nghia la anh dataset da duoc normalize

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            #print(text)
            target = [self.CHAR2LABEL[c] for c in text]
            #print(target)
    
            target_length = [len(target)]
            
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            
            return image, target, target_length
        else:
            return image
# get_item của dataset trả về image , index của text , len của text
