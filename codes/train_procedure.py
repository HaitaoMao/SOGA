import torch
import os
from torch import nn
import logging
from model.model_utilies import save_model, load_model, test, process_adj, generate_normalized_adjs, generate_one_hot_label, predict
import torch.nn.functional as F

def train_procedure(args, logger, model, source_optimizer, target_optimizer, criterion, source_data, target_data, target_structure_data, structure_adj):
    # training
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_val_loss = 1e8

    args.logger.info("source period")
    if args.is_source_train:
        for epoch in range(args.source_epochs):
            train_loss, train_accuracy = model.train_source(source_data, source_optimizer, criterion, epoch)
        
            val_loss, val_accuracy = test(model, args, source_data, criterion, 'valid')
            args.logger.info('Epoch\t{:03d}\ttrain:acc\t{:.6f}\tcross_entropy\t{:.6f}\tvalid:acc\t{:.6f}\tcross_entropy\t{:.6f}'.format(
            epoch, train_accuracy, train_loss, val_accuracy, val_loss))
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_loss = val_loss
                save_model(args, "source", model)
        args.logger.info('Best valid acc\t{:.6f}\t Best valid loss\t{:.6f}'.format(best_val_acc, best_val_loss))
    else:
        model = load_model(args, "source", model)


    args.logger.info("target period")
    if args.is_baseline:
        best_test_acc = test(model, args, target_data, criterion, "test")
    else:
        model = load_model(args, "source", model)
        model.init_target(target_structure_data, target_data)

        for epoch in range(args.target_epochs):
            test_accuracy = model.train_target(target_data, target_structure_data,structure_adj, criterion, target_optimizer, epoch)
            args.logger.info('Epoch\t{:03d}\ttest:acc\t{:.6f}'.format(epoch, test_accuracy))
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                save_model(args, "target", model)
            
    args.logger.info('Best test acc\t{:.6f}'.format(best_test_acc))


    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    return best_test_acc
