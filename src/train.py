data_dir = './data'
train_loader, val_loader = get_voc_dataloaders(data_dir, batch_size=4, num_workers=0) 