import os
import torch
import torchvision.transforms as transforms
import VideoFrameDataset
import torch.optim as optim
import numpy as np
from CustomHighPassFilters import get_high_pass_filters
from HelperFunctions import check_cuda, get_frame_dataset, load_ckp
from NoiseResidualCNN import CNN_Steganalysis
from Settings import *
from torchvision.utils import save_image


def eval_model(model, loaders, use_cuda, train_or_test):
    print("Evaluating model on " + train_or_test + " set...")

    # set model to eval mode
    model.eval()

    table = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    test_acc = 0.0
    total_images = 0
    f = 1
    for batch_idx, (data_eval, target_eval) in enumerate(loaders[train_or_test]):
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

        # get predictions
        pred = torch.argmax(output_eval, dim=1)

        for i in range(len(pred)):
            table[target_eval.data[i]][pred.data[i]] += 1

            if target_eval.data[i] != pred.data[i]:
                img = data_eval[i]  # torch.Size([3,28,28]
                # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
                save_image(img, './class_error/img'+train_or_test+str(f)+'_class_'+str(target_eval.data[i].item())+'_pred_'+str(pred.data[i].item())+'.png')
                f += 1

        # update accuracy
        correct = pred.eq(target_eval)
        test_acc += torch.mean(correct.float())

        total_images += pred.size()[0]

    # print training/validation statistics
    print('Accuracy of the network on {} {} images: {}%'.format(
        total_images,
        train_or_test,
        round(test_acc.item() * 100.0 / len(loaders[train_or_test]), 2)
    ))

    print(np.matrix(table))


def main():
    # Initialise model
    model = CNN_Steganalysis()

    # check if CUDA is available
    use_cuda = check_cuda()
    if use_cuda:
        # Initialise model
        model = model.cuda()
        # Set custom filters ResConv
        model.ResConv.weight.data = torch.Tensor(get_high_pass_filters()).cuda()

    # Set test settings
    batch_size = 1
    num_segments = 20
    frames_per_segment = 1
    optimizer = optim.Adadelta(model.parameters(), lr=0.4, weight_decay=5e-4, rho=0.95, eps=1e-8)

    # Create path variables
    root = os.path.join(os.getcwd(), dir_CNN_dataset)
    annotation_file_test = os.path.join(root, test_annotations)
    annotation_file_train = os.path.join(root, train_annotations)

    # List of PIL images to (FRAMES x 1 x 224 x 224) tensor
    pre_process = transforms.Compose([
        VideoFrameDataset.resizeList((224, 224)),
        VideoFrameDataset.grayscale_list(1),
        VideoFrameDataset.ImglistToTensor()
    ])

    test_set = get_frame_dataset(root, annotation_file_test, num_segments, frames_per_segment, pre_process)
    train_set = get_frame_dataset(root, annotation_file_train, num_segments, frames_per_segment, pre_process)

    loaders = {
        'test': torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        ),
        'train': torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        ),
    }

    model, optimizer, start_epoch, valid_loss_min = load_ckp("best_model/best_model.pt", model, optimizer)

    eval_model(model, loaders, use_cuda, "train")
    eval_model(model, loaders, use_cuda, "test")


if __name__ == '__main__':
    main()
