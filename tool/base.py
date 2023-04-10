from lib import *
from torch.nn import CTCLoss
from data.synthdataset import Synth90kDataset, synth90k_collate_fn
from data.icdardataset import Icdar15Dataset
from utils.model.crnn import CRNN
from utils.evaluate import evaluate,predict
#from tool.test import predict, get_images
from utils.model.ctc_decoder import ctc_decode
from utils.pre_processing.img_proc import denormalization, create_dir, get_images,txt_into_list
from utils.callback.scheduler import CosineAnnealingWarmupRestarts
from utils.metrics.metric_monitor import MetricMonitor



class TrainSynth90k(object):
    def __init__(self, model):
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(self.device)
        self.metric_monitor = MetricMonitor()


    def save_model(self, path, step, model, optimizer, scheduler,best_score):
        state_dict = model.state_dict()

        torch.save({
            'step': step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score,
        }, path)


    def load_model(self, model, optimizer,scheduler ,resume):
        checkpoint = torch.load(resume, map_location=self.device)
        #print(checkpoint.keys())
        print('loaded weights from {}, step {}, best_score {}'.format(
            resume, checkpoint['step'], checkpoint['best_score']))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        step = checkpoint['step']
        best_score = checkpoint['best_score']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return model, optimizer, step,scheduler ,best_score
    
    
    def train_network(self, cfg):
        """ Seeding """
        #seeding(cfg['SEED'])

    

        """ Savedir """
        cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{cfg['DATASET']['target']}" # CRAFT-synht_icdar15
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME']) #saved_model/CRAFT-synht_icdar15
        vis_test_dir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'], cfg['RESULT']['vis_test_dir']) # saved_model/CRAFT-synht_icdar15/vis_test_dir #  cfg['EXP_NAME'] = CRAFT-synht_icdar15
        create_dir(savedir)
        create_dir(vis_test_dir)

         # wandb
        api_key = cfg['API_KEY']
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()

        if cfg['TRAIN']['use_wandb']:
            wandb.init(name=cfg['EXP_NAME'], project='CRNN', config=cfg)




        """ Date & Time """
        datetime_object = str(datetime.datetime.now())
        print(datetime_object)
        print("")
        print(cfg['RESUME']['bestmodel'])
        """ Hyperparameters """
        data_str = f"Image Size: {cfg['DATASET']['resize']['height']} {cfg['DATASET']['resize']['width']}\nBatch Size: {cfg['TRAIN']['train_batch_size']}\nLR: {cfg['OPTIMIZER']['lr']}\nSteps: {cfg['TRAIN']['num_training_steps']}\n"
        print(data_str)



        num_class = len(Synth90kDataset.LABEL2CHAR) + 1

        self.model = self.model(1, cfg['DATASET']['resize']['height'], cfg['DATASET']['resize']['width'], num_class,
                map_to_seq_hidden=cfg['MODEL']['map_to_seq_hidden'],
                rnn_hidden=cfg['MODEL']['rnn_hidden'],
                leaky_relu=cfg['MODEL']['leaky_relu'])

        print(
            "Model A.D. Param#: {}".format(
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            )
        )

        wandb.watch(self.model)

        if os.path.exists(os.path.join(savedir, cfg['RESUME']['bestmodel'])):
            print("loadddd")
            checkpoint = torch.load(os.path.join(savedir, cfg['RESUME']['bestmodel']), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])            


        # Set training
        # Opimizer
        if cfg['OPTIMIZER']['optim_name'] == 'adam':
            self.optimizer = torch.optim.AdamW(
                params       = filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr           = cfg['OPTIMIZER']['lr'], 
                weight_decay = cfg['OPTIMIZER']['weight_decay']
            )
            optimizer_name = "AdamW"


        else:
            self.optimizer = torch.optim.SGD(
                params       = filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr           = cfg['OPTIMIZER']['lr'], 
                momentum     = cfg['OPTIMIZER']['momentum'],
                weight_decay = cfg['OPTIMIZER']['weight_decay']
            )
            optimizer_name = "SGD"
        
        
        if cfg['SCHEDULER']['use_scheduler']:
            scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer, 
                first_cycle_steps   = cfg['TRAIN']['num_training_steps'],
                max_lr              = cfg['OPTIMIZER']['lr'],
                min_lr              = cfg['SCHEDULER']['min_lr'],
                warmup_steps        = int(cfg['TRAIN']['num_training_steps'] * cfg['SCHEDULER']['warmup_ratio'])
            )
        else:
            scheduler = None

        # loss
        CTC_criterion = CTCLoss(reduction='sum', zero_infinity=True)
        loss_name = "CTC Loss"
        CTC_criterion.to(self.device)

        data_str = f"Optimizer: {optimizer_name}\nLoss: {loss_name}\n"
        print(data_str)

        print('Setting up data...')
      
        train_synth_dataset = Synth90kDataset(
            root_dir    = 'text-reg-data/base-data/mnt/ramdisk/max/90kDICT32px',
            mode        = 'train',
            img_height  = cfg['DATASET']['resize']['height'], 
            img_width   = cfg['DATASET']['resize']['width'] )
        
        print(len(train_synth_dataset))
    
        train_special_character_dataset = Icdar15Dataset(
            root_dir    = 'text-reg-data/base-data/special_character10',
            mode        = 'train_aug',
            img_height  = cfg['DATASET']['resize']['height'], 
            img_width   = cfg['DATASET']['resize']['width'])
        
        print(len(train_special_character_dataset))

        train_anotherexp_dataset = Icdar15Dataset(
            root_dir    = 'text-reg-data/base-data/anotherexp',
            mode        = 'train_anotherexp',
            img_height  = cfg['DATASET']['resize']['height'], 
            img_width   = cfg['DATASET']['resize']['width'])
        print(len(train_anotherexp_dataset))
        trainset = ConcatDataset([train_synth_dataset, train_special_character_dataset, train_anotherexp_dataset])

        train_loader = DataLoader(
                    dataset     = trainset,
                    batch_size  = cfg['TRAIN']['train_batch_size'],
                    shuffle     = True,
                    num_workers = cfg['TRAIN']['num_workers'],
                    collate_fn=synth90k_collate_fn)


        valid_synth_dataset = Synth90kDataset(
            root_dir    = 'text-reg-data/base-data/mnt/ramdisk/max/90kDICT32px', 
            mode        = 'dev', # chọn tập valid
            img_height  = cfg['DATASET']['resize']['height'], 
            img_width   = cfg['DATASET']['resize']['width'])
        print(len(valid_synth_dataset))
        valid_special_character_dataset = Icdar15Dataset(
            root_dir    = 'text-reg-data/base-data/special_character10',
            mode        = 'val_aug',
            img_height  = cfg['DATASET']['resize']['height'], 
            img_width   = cfg['DATASET']['resize']['width'])
        print(len(valid_special_character_dataset))
        valid_anotherexp_dataset = Icdar15Dataset(
            root_dir    = 'text-reg-data/base-data/anotherexp',
            mode        = 'val_anotherexp',
            img_height  = cfg['DATASET']['resize']['height'], 
            img_width   = cfg['DATASET']['resize']['width'])
        print(len(valid_anotherexp_dataset))
        validset = ConcatDataset([valid_synth_dataset,valid_special_character_dataset ,valid_anotherexp_dataset])


        valid_loader = DataLoader(
            dataset     = validset,
            batch_size  = cfg['EVALUATE']['eval_batch_size'],
            shuffle     = False,
            num_workers = cfg['EVALUATE']['num_workers'],
            collate_fn  = synth90k_collate_fn)


        #images_path_list = get_images(cfg['TEST']['IMAGE'])
        images_synth_path_list = txt_into_list(cfg)[:500] # truyền vào annotation test file thì nó tự tự láy ra path file 
        
        images_special_character_path_list = get_images('text-reg-data/base-data/special_character10')[800:1000]
        images_anotherexp_path_list = get_images('text-reg-data/base-data/anotherexp')
        images_path_list = images_synth_path_list + images_special_character_path_list  + images_anotherexp_path_list
        random.shuffle(images_path_list)
        predict_dataset = Synth90kDataset(
                    paths = images_path_list,
                    img_height=cfg['DATASET']['resize']['height'], 
                    img_width=cfg['DATASET']['resize']['width'])
      
      
        data_str = f"Dataset Size:\nTrain: {len(trainset)} - Valid: {len(validset)}\n - Predict: {len(predict_dataset)}\n"
        print(data_str)

        print('Visualize augmentations...')
        self.visualize_augmentations(trainset, samples=10)


        # Fitting model
        self.training(
            num_training_steps  = cfg['TRAIN']['num_training_steps'], 
            train_loader        = train_loader, 
            valid_loader        = valid_loader, 
            predict_dataset     = predict_dataset, 
            criterion           = CTC_criterion, 
            scheduler           = scheduler,
            valid_decode_method = cfg['EVALUATE']['decode_method'],
            valid_beam_size     = cfg['EVALUATE']['beam_size'],
            log_interval        = cfg['LOG']['log_interval'],
            eval_interval       = cfg['LOG']['eval_interval'],
            savedir             = savedir,
            resume              = cfg['RESUME']['bestmodel'],
            use_wandb=cfg['TRAIN']['use_wandb']           
        )

        wandb.save(os.path.join(savedir, 'best_score.json'),
                       base_path=savedir)
    def training(self, train_loader, valid_loader,predict_dataset, criterion, scheduler,valid_decode_method,valid_beam_size, num_training_steps: int = 1000, 
                log_interval: int = 1, eval_interval: int = 1, savedir: str = None, resume: str = None,use_wandb: bool = False) -> dict:
        print('Starting training...')
        print('-'*10)
    

        best_score = 0
        step = 0
        train_mode = True

        print(f"resume : {os.path.join(savedir, resume)}")

        if os.path.exists(os.path.join(savedir, resume)):
            print("LOAD MODEL")
            self.model, self.optimizer, step, scheduler,best_score = self.load_model(model = self.model,optimizer = self.optimizer,scheduler = scheduler,resume = os.path.join(savedir, resume))                                          



        self.model.train() # model này năm ở init
        self.optimizer.zero_grad()

        while train_mode:
            stream = tqdm(train_loader)

            
            self.model.to(self.device)

            loss = criterion
            
            
            for train_data in train_loader: # 1 batchsize train_data = 1 step

                loss = self.train_batch(self.model, train_data, self.optimizer, criterion, self.device)
                self.metric_monitor.update("Loss", loss, self.optimizer.param_groups[0]['lr']) #( metric_name, loss, lr)
# update loss

                if use_wandb:
                    wandb.log({
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'loss' : loss},
                        step=step)


                if (step+1) % log_interval == 0 or step == 0:
                    
                    stream.set_description(
                        "TRAIN [{:>4d}/{}].     {metric_monitor}".format(
                            step+1, num_training_steps, metric_monitor=self.metric_monitor)
                    ) # gội hàm __str__ của AverageMetric

                

                if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps:
                    evaluation = evaluate(self.model, valid_loader, criterion,
                                      decode_method=valid_decode_method,
                                      beam_size=valid_beam_size)
                    preds = predict(self.model, predict_dataset, Synth90kDataset.LABEL2CHAR,
                    decode_method='beam_search',
                    beam_size=10)

                    self.model.train()

                    eval_log = dict([(f'eval_{loss}', acc) for loss, acc in evaluation.items()])
                    # checkpoint
                    print(f"acc : {evaluation['acc']}")
                    
                    if best_score < evaluation['acc']:# chỗ này phải > thì mới lưu lại chứ ta
                        best_score = evaluation['acc']

                        # save best score
                        state = {'best_step':step}
                        state.update(eval_log)

                        if use_wandb:
                            wandb.log(eval_log, step=step)


                        json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                        print("SAVE_MODEL")
                        #  save best model
                        self.save_model(os.path.join(savedir, resume), # checkpoint save theo hmean
                                step,
                                self.model,
                                self.optimizer,
                                scheduler,
                                best_score)

                # scheduler
                if scheduler:
                    scheduler.step()

                step += 1

                if step == num_training_steps:
                    train_mode = False
                    break


    def train_batch(self,crnn, data, optimizer, criterion, device):
        crnn.train()
        images, targets, target_lengths = [d.to(device) for d in data]

        logits = crnn(images) # đưa qua model 
        log_probs = torch.nn.functional.log_softmax(logits, dim=2) 

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size) # ??? chỗ này là sao
        target_lengths = torch.flatten(target_lengths)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5 ???
        optimizer.step() # tính toán gradient (update gradient)
        optimizer.zero_grad() # empty gradient
        
        
       
        return loss.item()
    

    def visualize_augmentations(self, dataset, samples=10):
        #dataset = copy.deepcopy(dataset)
        n = np.arange(0, len(dataset))
        figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(15, 10))
        for i in range(samples):
            image, target, _ = dataset[np.random.choice(n)]
            #print(type(image))
            text = [Synth90kDataset.LABEL2CHAR[c] for c in target.cpu().numpy()]
            ax[i, 0].imshow(image.numpy().transpose(1, 2, 0),
                            interpolation="nearest", cmap='gray')
            ax[i, 0].set_title("Text: {}".format(text))
            ax[i, 0].set_axis_off()
            ax[i, 1].imshow(image.numpy().transpose(1, 2, 0),
                            interpolation="nearest", cmap='gray')
            ax[i, 1].set_title("Text: {}".format(target.cpu().numpy()))
            ax[i, 1].set_axis_off()
            

        plt.tight_layout()
        plt.show()





