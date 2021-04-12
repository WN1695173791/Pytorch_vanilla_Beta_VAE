import torch
from dataset.dataset_2 import get_mnist_dataset
from viz.visualizer_functions import model_load

# parameters:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_test = torch.load('../data/batch_mnist.pt')
model_path = '../checkpoints_CNN/'
batch_size = 128  # 10.000 to run regions visualization (we must have only one so all data set test in one batch)
train_loader, test_loader = get_mnist_dataset(batch_size=batch_size)
img_size = (1, 32, 32)
nb_class = 10
nc = 1

parameters_mnist_classifier_BK_ratio = "parameters_combinations/mnist_classifier_ratio.txt"

# Encoder struct list experimentation:
list_encoder_struct = ['mnist_struct_baseline_scheduler_binary_10']
                       # 'mnist_struct_baseline_scheduler_binary_15',
                       # 'mnist_struct_baseline_scheduler_binary_20',
                       # 'mnist_struct_baseline_scheduler_binary_25',
                       # 'mnist_struct_baseline_scheduler_binary_30']

list_encoder_struct_Hmg = ['mnist_struct_baseline_scheduler_binary_10_Hmg_dst_1',
                           'mnist_struct_baseline_scheduler_binary_15_Hmg_dst_1',
                           'mnist_struct_baseline_scheduler_binary_20_Hmg_dst_1',
                           'mnist_struct_baseline_scheduler_binary_25_Hmg_dst_1',
                           'mnist_struct_baseline_scheduler_binary_30_Hmg_dst_1',
                           'mnist_struct_baseline_scheduler_binary_10_target_uc_1',
                           'mnist_struct_baseline_scheduler_binary_15_target_uc_1',
                           'mnist_struct_baseline_scheduler_binary_20_target_uc_1',
                           'mnist_struct_baseline_scheduler_binary_25_target_uc_1',
                           'mnist_struct_baseline_scheduler_binary_30_target_uc_1',
                           'mnist_struct_baseline_scheduler_binary_10_Hmg_dst_2',
                           'mnist_struct_baseline_scheduler_binary_15_Hmg_dst_2',
                           'mnist_struct_baseline_scheduler_binary_20_Hmg_dst_2',
                           'mnist_struct_baseline_scheduler_binary_25_Hmg_dst_2',
                           'mnist_struct_baseline_scheduler_binary_30_Hmg_dst_2',
                           'mnist_struct_baseline_scheduler_binary_10_target_uc_2',
                           'mnist_struct_baseline_scheduler_binary_15_target_uc_2',
                           'mnist_struct_baseline_scheduler_binary_20_target_uc_2',
                           'mnist_struct_baseline_scheduler_binary_25_target_uc_2',
                           'mnist_struct_baseline_scheduler_binary_30_target_uc_2',
                           'mnist_struct_baseline_scheduler_binary_10_Hmg_dst_3',
                           'mnist_struct_baseline_scheduler_binary_15_Hmg_dst_3',
                           'mnist_struct_baseline_scheduler_binary_20_Hmg_dst_3',
                           'mnist_struct_baseline_scheduler_binary_25_Hmg_dst_3',
                           'mnist_struct_baseline_scheduler_binary_30_Hmg_dst_3',
                           'mnist_struct_baseline_scheduler_binary_10_target_uc_3',
                           'mnist_struct_baseline_scheduler_binary_15_target_uc_3',
                           'mnist_struct_baseline_scheduler_binary_20_target_uc_3',
                           'mnist_struct_baseline_scheduler_binary_25_target_uc_3',
                           'mnist_struct_baseline_scheduler_binary_30_target_uc_3']

if __name__ == '__main__':

    # load model from experimentation list:
    for model_name in list_encoder_struct:
        mdoel = model_load(model_name,
                           model_path=model_path,
                           device=device,
                           verbose=True)

    for model_name in list_encoder_struct_Hmg:
        mdoel = model_load(model_name,
                           model_path=model_path,
                           device=device,
                           verbose=True)
