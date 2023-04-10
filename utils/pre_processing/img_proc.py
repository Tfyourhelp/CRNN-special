from lib import *

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x * std) + mean) * 255.)
    return x


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_images(data_path):
    '''
    tìm tệp hình ảnh trong đường dẫn
    return: danh sách tệp được tìm thấy
    '''
    files = []
    idx = 0
    for ext in ['jpg','png']:
        files.extend(glob.glob(
            os.path.join(data_path, '*.{}'.format(ext))))
        idx += 1
    return files


def txt_into_list(cfg):
    lines = []
    filename = cfg['TEST']['ANNOTATION_TEST']
    with open(filename) as file:
        # i = 0
        for line in file:
            line = line.split(' ')[0]
            line = line[1:]
            # print(line)
            line = 'text-reg-data/base-data/mnt/ramdisk/max/90kDICT32px' + line
            lines.append(line)
            # i+=1
            # if i == 100:
            #     break
    return lines

def processing_img(image, width, height, cval = 255, mode = "letterbox", return_scale = False):
    fitted = None
    x_scale = width / image.shape[1]  # h,w of input image
    y_scale = height / image.shape[0]

    if x_scale == 1 and y_scale == 1:
        fitted = image
        scale = 1
    elif (x_scale <= y_scale and mode == "letterbox") or (
        x_scale >= y_scale and mode == "crop"
    ):
        scale = width / image.shape[1] # width of input image
        resize_width = width
        resize_height = (width / image.shape[1]) * image.shape[0]
    else:
        scale = height / image.shape[0]
        resize_height = height
        resize_width = scale * image.shape[1]
    if fitted is None:
        resize_width, resize_height = map(int, [resize_width, resize_height])
        if mode == "letterbox":
            #fitted = np.zeros((height, width, 3), dtype="uint8") + cval
            fitted = np.zeros((height, width), dtype="uint8") + cval
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted[: image.shape[0], : image.shape[1]] = image[:height, :width]
        elif mode == "crop":
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted = image[:height, :width]
    if not return_scale:
        return fitted
    return fitted, scale