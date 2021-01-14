import tensorflow as tf
from hyper_resilient_experiments.segmentation.gis_preprocess import (tf_gis_test_train_split, pt_gis_train_test_split,
                                                                     perturbed_tf_gis_test_data,
                                                                     perturbed_pt_gis_test_data,
                                                                     PT_GISDataset)
from hyper_resilient_experiments.segmentation import pytorch_unet, tensorflow_unet
from torch.utils.data import DataLoader
from pytorch_lightning.metrics import Accuracy
import statistics
import torch
from tqdm import tqdm

if __name__ == "__main__":
    test_config = {'batch_size': 4, 'learning_rate': .001, 'epochs': 1, 'adam_epsilon': 10 ** -9}
    # res = cityscapes_tf_objective(test_config)
    # print(res[0])
    # cityscapes_tf_objective(test_config)
    # res = tensorflow_unet.gis_tf_objective(test_config)
    # x_test, y_test = perturbed_tf_gis_test_data()
    # test_acc = res[1].evaluate(x_test, y_test, batch_size=test_config['batch_size'])
    # print(res[0])
    # print(test_acc)

    print("PyTorch Model evaluation...")
    acc, pt_model = pytorch_unet.gis_pt_objective(test_config)
    test = perturbed_pt_gis_test_data()
    test_set = PT_GISDataset(test)
    testloader = DataLoader(test_set, batch_size=int(test_config['batch_size']))
    accuracy = Accuracy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for sample in tqdm(testloader):
        cuda_in = sample[0].to(device)
        out = pt_model(cuda_in)
        label = sample[1].to(device)
        accuracy(out.squeeze(1), label)
        # cuda_in = cuda_in.detach()
        # label = label.detach()
    print("AVERAGE ACCURACY:")
    print(accuracy.compute())
