import torch
import torch.nn as nn
import torch.nn.parameter
import torchvision.transforms as transforms
import torch.optim as optim
import VideoFrameDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from CustomHighPassFilters import get_high_pass_filters
from HelperFunctions import check_cuda, save_ckp, get_frame_dataset
from NoiseResidualCNN import CNN_Steganalysis
from VideoFrameDataset import VideoFrameDataset
from Settings import *
# from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def vis_filter(filters):
    plt.figure(figsize=(20, 17))
    for i, f in enumerate(filters):
        plt.subplot(6, 6, i + 1)
        plt.imshow(f[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()


def train_model(model, loaders, use_cuda, optimizer, criterion, train_loss, train_acc):
    model.train()
    print("Training model...")
    for batch_idx, (data, target) in enumerate(loaders['train']):
        # transform input in right format
        targetArray = []
        for t in target:
            for i in range(data.size()[1]):
                targetArray.append(t.item())
        data = torch.reshape(data,
                             (data.size()[0] * data.size()[1], data.size()[2], data.size()[3], data.size()[4]))
        target = torch.tensor(targetArray, dtype=torch.long)

        # plot_video(rows=3, cols=5, frame_list=data, plot_width=30., plot_height=9., title=target.__str__())

        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # record the average training loss, using something like
        # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        # update accuracy
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target)
        train_acc += torch.mean(correct.float())

    return model, optimizer, train_loss, train_acc


def eval_model(model, loaders, use_cuda, criterion, train_loss, valid_loss, epoch, test_acc, train_acc):
    model.eval()
    print("Evaluating model...")
    for batch_idx, (data_eval, target_eval) in enumerate(loaders['validate']):
        # transform input in right format
        target_evalArray = []
        for t in target_eval:
            for i in range(data_eval.size()[1]):
                target_evalArray.append(t.item())
        data_eval = torch.reshape(data_eval,
                                  (data_eval.size()[0] * data_eval.size()[1], data_eval.size()[2],
                                   data_eval.size()[3], data_eval.size()[4]))
        target_eval = torch.tensor(target_evalArray, dtype=torch.long)
        # move to GPU
        if use_cuda:
            data_eval, target_eval = data_eval.cuda(), target_eval.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        with torch.no_grad():
            output_eval = model(data_eval)
        # calculate the batch loss
        loss = criterion(output_eval, target_eval)
        # update average validation loss
        valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        # update accuracy
        pred = torch.argmax(output_eval, dim=1)
        correct = pred.eq(target_eval)
        test_acc += torch.mean(correct.float())

    # calculate average losses
    train_loss = train_loss / len(loaders['train'])
    valid_loss = valid_loss / len(loaders['validate'])

    # print training/validation statistics
    print(
        'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTrain accuracy: {:.6f} \tTest accuracy: {:.6f} '.format(
            epoch,
            train_loss,
            valid_loss,
            round(train_acc.item() * 100.0 / len(loaders['train']), 2),
            round(test_acc.item() * 100.0 / len(loaders['validate']), 2)
        ))

    return model, valid_loss, test_acc, train_acc


def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, criterion, use_cuda, checkpoint_path,
          best_model_path):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path

    returns trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epochs, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        test_acc = 0.0
        train_acc = 0.0

        ###################
        # train the model #
        ###################
        model, optimizer, train_loss, train_acc = train_model(model, loaders, use_cuda, optimizer, criterion,
                                                              train_loss, train_acc)

        ######################
        # validate the model #
        ######################
        model, valid_loss, test_acc, train_acc = eval_model(model, loaders, use_cuda, criterion, train_loss, valid_loss,
                                                            epoch, test_acc, train_acc)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'test_acc': test_acc,
            'train_acc': train_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss

    # return trained model
    print("Finished training model...")
    return model


def main():
    # initialise model
    model = CNN_Steganalysis()
    # Set custom filters ResConv
    model.ResConv.weight.data = torch.Tensor(get_high_pass_filters())

    # check if CUDA is available
    use_cuda = check_cuda()
    if use_cuda:
        # Initialise model
        model = model.cuda()
    else:
        return

    # Print model summary
    # summary(model, (1, 224, 224))
    # Visualise filter first layer
    # vis_filter(model.ResConv.weight.data.clone().cpu())

    # Set training settings
    batch_size = 4
    num_epochs = 150
    num_segments = 5
    frames_per_segment = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.4, weight_decay=5e-4, rho=0.95, eps=1e-8)

    # Create path variables
    root = os.path.join(os.getcwd(), dir_CNN_dataset)
    annotation_file_train = os.path.join(root, train_annotations)
    annotation_file_validate = os.path.join(root, validate_annotations)
    annotation_file_test = os.path.join(root, test_annotations)

    # List of PIL images to (FRAMES x 1 x 224 x 224) tensor
    pre_process = transforms.Compose([
        VideoFrameDataset.resizeList((224, 224)),
        VideoFrameDataset.grayscale_list(1),
        VideoFrameDataset.ImglistToTensor()
    ])

    train_set = get_frame_dataset(root, annotation_file_train, num_segments, frames_per_segment, pre_process)
    validation_set = get_frame_dataset(root, annotation_file_validate, num_segments, frames_per_segment, pre_process)
    test_set = get_frame_dataset(root, annotation_file_test, num_segments, frames_per_segment, pre_process)

    loaders = {
        'train': torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'validate': torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'test': torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    }

    trained_model = train(1, num_epochs, np.Inf, loaders, model, optimizer, criterion, use_cuda,
                          "checkpoint/current_checkpoint.pt", "best_model/best_model.pt")


if __name__ == '__main__':
    main()
