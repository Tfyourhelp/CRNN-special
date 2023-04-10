from lib import *
from utils.pre_processing.img_proc import processing_img


class Synth90kDataset(Dataset):
    #CHARS = string.digits +  string.ascii_lowercase + string.ascii_uppercase

    CHARS = string.digits +  string.ascii_lowercase + string.ascii_uppercase + "-./:"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)} # char sang index
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()} # index sang char

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        
    
        if root_dir and mode and not paths: #nếu muốn dùng Synth thì truyền vào root_dir, không truyền vào path | còn muốn dùng để test thì chỉ truyền vào path không truyền vào root_dir,path
# có nghĩa là dùng để test (predict) thì chỉ truyền vào ảnh , dùng để train ,evaluate phải truyền vào ảnh và text để tính loss
            paths, texts = self._load_from_raw_files(root_dir, mode)
            
            if mode == 'train': # train thì train 10000 ảnh đầu tiên
                paths = paths[:100000]
                texts = texts[:100000]
            elif mode == 'dev':  # đánh giá thì 500 ảnh cuối cùng
                paths = paths[-5000:]
                texts = texts[-5000:]
            elif mode == 'test':  # đánh giá thì 10 ảnh kế cuối
                paths = paths[-5100:-5000]
                texts = texts[-5100:-5000]


        elif not root_dir and not mode and paths: # trường hợp để root_dir không có , mode không có
            texts = None
            # khúc này phải làm cách paths thành cái list
        #print(texts)
        self.paths = paths
        self.texts = texts
        # print('k')
        # print(self.paths)
        # print(self.texts)
        self.img_height = img_height
        self.img_width = img_width





    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

    
        paths = []
        texts = []
    
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
    
                path, index_str = line.strip().split(' ')
                path = path[2:]
                path = os.path.join(root_dir, path)
                index = int(index_str) # không cần sử dụng 
                #text = mapping[index]
                #print(path)
                text = path.split('_')[1]
               
                paths.append(path)
                texts.append(text)
                
        return paths, texts


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        #print(path)
        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print(path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = np.array(image)
            
    
        #image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = processing_img(image, self.img_width, self.img_height, cval = 0)
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
            # print(target_length)
            # print('\n')
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    
    images = torch.stack(images, 0) 
    targets = torch.cat(targets, 0) 
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths