from lib import *
from utils.pre_processing.img_proc import denormalization, create_dir
from data.synthdataset import Synth90kDataset, synth90k_collate_fn
from data.icdardataset import Icdar15Dataset
from utils.model.ctc_decoder import ctc_decode
from utils.model.crnn import CRNN
from configs.config import evaluate_config as config

torch.backends.cudnn.enabled = False
#CHARS = string.digits +  string.ascii_lowercase + string.ascii_uppercase
CHARS = string.digits +  string.ascii_lowercase + string.ascii_uppercase + "-./:"
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)} 
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}


def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader) # vẽ thanh bar
    pbar = tqdm(total=pbar_total, desc="Evaluate") # vẽ thanh bar

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()
            #print("".join(LABEL2CHAR[l] for l in reals))
            #print(LABEL2CHAR[l] for l in preds)
            # print("".join(LABEL2CHAR[l] for l in reals))
            # print("".join(LABEL2CHAR[l] for l in preds))
            # print('\n')
            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation

def predict(crnn, dataset, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataset), desc="Predict")
    images = []
    all_preds = []
    i = 0

    with torch.no_grad():
        
        for i in range(0, 16):
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            
            index = np.random.choice(len(dataset)) 
            

            image = torch.unsqueeze(dataset[index],0) # trong dataset nó đã chuyển sang tensor rồi nên kh cần phải from numpy to tensor
            image = image.to(device)#--> 'DataLoader' object does not support indexing ??

            logits = crnn(image)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            
            all_preds += preds
            images.append(dataset[index])

            pbar.update(1)
        pbar.close()

        fig , ax = plt.subplots(4, 4, figsize = (15, 5))
        #fig.suptitle('Epoch: ' + str(int(epoch) + 1), weight='bold', size=14)

        image1 = [image.numpy().transpose(1, 2, 0) for image in images]
        image1 = denormalization(image1)
       
      
        for i in range(16):
            a = ''.join(all_preds[i])
            title = f"Prediction: {a}"
            ax[i // 4, i % 4].imshow(image1[i].astype("uint8"))
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
            
        plt.savefig('saved_model/CRNN-mjsynth/vis_test_dir/books_read.png')
        plt.show()


    return all_preds


def main():
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    test_dataset = Synth90kDataset(root_dir=config['data_dir'], mode='test',
                                   img_height=img_height, img_width=img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()