from lib import *
from docopt import docopt
from data.synthdataset import Synth90kDataset, synth90k_collate_fn
from data.icdardataset import Icdar15Dataset
from utils.model.crnn import CRNN
from utils.model.ctc_decoder import ctc_decode
#from utils.pre_processing.img_proc import get_images
from utils.pre_processing.img_proc import denormalization, create_dir, get_images,txt_into_list
from utils.pre_processing.img_proc import processing_img


# def predict(crnn, dataloader, label2char, decode_method, beam_size):
#     crnn.eval()
#     pbar = tqdm(total=len(dataloader), desc="Predict")

#     all_preds = []
#     with torch.no_grad():
#         for data in dataloader:
#             device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

#             images = data.to(device)

#             logits = crnn(images)
#             log_probs = torch.nn.functional.log_softmax(logits, dim=2)

#             preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
#                                label2char=label2char)
#             all_preds += preds

#             pbar.update(1)
#         pbar.close()

#     return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')



def get_images_and_path(paths,img_height, img_width):
    images = []
    image2s = []
    for path in paths:
    
        image = Image.open(path).convert('L')  # grey-scale
      
        image = np.array(image)
        
        #image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = processing_img(image, img_width, img_height, cval = 0)
        #print(image.shape)

        image = image.reshape((1, img_height, img_width)) #?? chỗ này là sao
        image = (image / 127.5) - 1.0 #normalize --> co nghia la anh dataset da duoc normalize

        image = torch.FloatTensor(image)
        images.append(image)
        
    return images,paths


# in ra không màu
# def predict(crnn, dataset, label2char, decode_method, beam_size):
#     crnn.eval()
#     pbar = tqdm(total=len(dataset), desc="Predict")
#     images = []
#     all_preds = []
#     i = 0

#     with torch.no_grad():
#         #for data in dataloader: # predict hết cả dataloader
#         for i in range(0, 16):
#             device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            
#             index = np.random.choice(len(dataset)) 
            
#             image = torch.unsqueeze(dataset[index],0) # trong dataset nó đã chuyển sang tensor rồi nên kh cần phải from numpy to tensor
#             image = image.to(device)#--> 'DataLoader' object does not support indexing ??

#             logits = crnn(image)
#             log_probs = torch.nn.functional.log_softmax(logits, dim=2)

#             preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
#                                label2char=label2char)
            
#             all_preds += preds
#             images.append(dataset[index])

#             pbar.update(1)
#         pbar.close()

#         fig , ax = plt.subplots(4, 4, figsize = (15, 5))
      
#         image1 = [image.numpy().transpose(1, 2, 0) for image in images]
#         image1 = denormalization(image1)
     
#         for i in range(16):
#             a = ''.join(all_preds[i])
#             title = f"Prediction: {a}"
#             ax[i // 4, i % 4].imshow(image1[i].astype("uint8"))
#             ax[i // 4, i % 4].set_title(title)
#             ax[i // 4, i % 4].axis("off")
            
#         plt.savefig('saved_model/CRNN-mjsynth/vis_test_dir/books_read.png')
#         plt.show()


#     return all_preds

# in ra có màu
def predict(crnn, dataset,paths, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataset), desc="Predict")
    image1s = []
    all_preds = []
    images = []
    i = 0

    with torch.no_grad():
        #for data in dataloader: # predict hết cả dataloader
        for i in range(0, 24):
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            
            index = np.random.choice(len(dataset)) 


            #image = data.to(device)
            path = paths[index]
            image1 = Image.open(path)
          

            image = torch.unsqueeze(dataset[index],0) # trong dataset nó đã chuyển sang tensor rồi nên kh cần phải from numpy to tensor
            image = image.to(device)#--> 'DataLoader' object does not support indexing ??

            logits = crnn(image)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            
            all_preds += preds

            images.append(dataset[index])


            image1s.append(image1)
            pbar.update(1)
        pbar.close()

     
        fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(15, 20))


        for i in range(24):
            a = ''.join(all_preds[i])
            title = f"Prediction: {a}"
            ax[i // 4, i % 4].imshow(image1s[i])
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
            
     
        plt.show()


    return all_preds

def main(cfg):
    arguments = docopt(__doc__)
#text-reg-data/New folder
#C:/Users/DELL/Desktop/image/hsd_aug # data hsd valid
    images_path_list = get_images('text-reg-data/icdar/icdar15/trainning') # D:/data/TextReg/Challengeee
    print(len(images_path_list))
    
   
    batch_size = int(cfg['TEST']['test_batch_size'])
    decode_method = cfg['TEST']['decode_method']
    beam_size = int(cfg['TEST']['beam_size'])

    img_height = cfg['DATASET']['resize']['height']
    img_width = cfg['DATASET']['resize']['width']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')


    #predict_dataset = Synth90kDataset(paths = images_path_list ,img_height=img_height, img_width=img_width)

    #predict_loader = DataLoader(
        # dataset=predict_dataset,
        # batch_size=batch_size,
        # shuffle=False)
    predict_dataset,paths = get_images_and_path(images_path_list,img_height=img_height, img_width=img_width)


    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=cfg['MODEL']['map_to_seq_hidden'],
                rnn_hidden=cfg['MODEL']['rnn_hidden'],
                leaky_relu=cfg['MODEL']['leaky_relu'])
    #C:/Users/DELL/Downloads/best_model.pth
    #best_model.pth
# best_model_24.pth # dự đoán hsd , icdar 13
#C:/Users/DELL/Downloads/basehsdicdar15/crnnicdarfordetectbaocaoicdar13/best_model_26_.pth # dự đoán icdar 13
    #crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device),strict = False)
    
    checkpoint = torch.load('C:/Users/DELL/Downloads/basehsdicdar15/crnnicdarfordetectbaocaoicdar13/best_model_26_.pth', map_location=device)
    crnn.load_state_dict(checkpoint['model_state_dict'])
    #crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device),strict = False)
    
    crnn.to(device)

    # preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
    #                 decode_method=decode_method,
    #                 beam_size=beam_size)

    # preds = predict(crnn, predict_dataset, Synth90kDataset.LABEL2CHAR,
    #                 decode_method=decode_method,
    #                 beam_size=beam_size)


    preds = predict(crnn, predict_dataset,paths , Synth90kDataset.LABEL2CHAR,
                    decode_method='beam_search',
                    beam_size=10)

    show_result(images_path_list, preds)


if __name__ == '__main__':
    cfg = yaml.load(open('configs/config.yaml','r'), Loader=yaml.FullLoader)
    main(cfg)