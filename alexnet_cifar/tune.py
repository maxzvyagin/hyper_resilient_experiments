from hyperspace import create_hyperspace
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
import foolbox as fb
import pytorch_lightning as pl
from tensorflow_alexnet import TensorFlow_AlexNet
from pytorch_alexnet import PyTorch_AlexNet
from mxnet_alexnet import mxnet_objective,
import statistics
from tqdm import tqdm
from argparse import ArgumentParser

def tf_objective(config):
    model = TensorFlow_AlexNet(config)
    model.fit()
    accuracy = model.test()
    return accuracy, model

def pt_objective(config):
    model = PyTorch_AlexNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True)
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model

def mx_objective(config):
    return mxnet_objective(config)

def model_attack(model, model_type, attack_type, config):
    if model_type == "pt":
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    else:
        fmodel = fb.TensorFlowModel(model, bounds=(0, 1))
    images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=config['batch_size'])
    if attack_type == "uniform":
        attack = fb.attacks.L2AdditiveUniformNoiseAttack()
    elif attack_type == "gaussian":
        attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
    elif attack_type == "saltandpepper":
        attack = fb.attacks.SaltAndPepperNoiseAttack()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    if model_type == "pt":
        robust_accuracy = 1 - success.cpu().numpy().astype(float).flatten().mean(axis=-1)
    else:
        robust_accuracy = 1 - success.numpy().astype(float).flatten().mean(axis=-1)
    return robust_accuracy


def multi_train(config):
    pt_test_acc, pt_model = tf_objective(config)
    tf_test_acc, tf_model = pt_objective(config)
    # now run attacks
    search_results = {'pt_test_acc': pt_test_acc, 'tf_test_acc': tf_test_acc}
    for attack_type in ['uniform', 'gaussian', 'saltandpepper']:
        for model_type in ['pt', 'tf']:
            if model_type == 'pt':
                acc = model_attack(pt_model, model_type, attack_type, config)
            else:
                acc = model_attack(tf_model, model_type, attack_type, config)
            search_results[model_type + "_" + attack_type + "_" + "accuracy"] = acc
    all_results = list(search_results.values())
    average_res = float(statistics.mean(all_results))
    search_results['average_res'] = average_res
    tune.report(**search_results)
    return search_results


if __name__ == "__main__":
    parser = ArgumentParser("Run AlexNet cross framework tuning on PyTorch and AlexNet.")
    parser.add_argument("--out","-o", help="Specify the out csv filename.", required=True)
    args = parser.parse_args()
    # Defining the hyperspace
    hyperparameters = [(0.00001, 0.1),  # learning_rate
                       (0.2, 0.9),  # dropout
                       (10, 100),  # epochs
                       (10, 1000)]  # batch size
    space = create_hyperspace(hyperparameters)
    # Perform runs and aggregate results
    results = []
    for section in tqdm(space):
        # create a skopt gp minimize object
        optimizer = Optimizer(section)
        search_algo = SkOptSearch(optimizer, ['learning_rate', 'dropout', 'epochs', 'batch_size'],
                                  metric='average_res', mode='max')
        analysis = tune.run(multi_train, search_alg=search_algo, num_samples=50, resources_per_trial={'gpu': 1})
        results.append(analysis)

    all_pt_results = results[0].results_df
    for i in range(1, len(results)):
        all_pt_results = all_pt_results.append(results[i].results_df)

    all_pt_results.to_csv(args.out)