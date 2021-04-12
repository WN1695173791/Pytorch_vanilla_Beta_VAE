from viz.visualizer import *
from models.custom_CNN_BK import Custom_CNN_BK


def run_score(exp_name, net):
    print(exp_name)
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net_trained, _, nb_epochs = get_checkpoints(net, path, exp_name)
    # scores and losses:
    plot_scores_and_loss_CNN(net_trained, exp_name, path_scores, save=True)

    return


# binary model:
list_exp_name_binary = ['CNN_mnist_custom_BK_2layer_bk1_20_binary_1',
                        'CNN_mnist_custom_BK_2layer_bk1_20_binary_2',
                        'CNN_mnist_custom_BK_2layer_bk2_20_binary_1',
                        'CNN_mnist_custom_BK_2layer_bk2_20_binary_2']

f = open("parameters_combinations/mnist_classifier_binary_exp.txt", "r")
arguments = {}
i = 0

i = 0
key = 0
for x in f:
    i += 1
    if i < 0:
        pass
    elif x[0].isspace():
        break
    elif x[0] != "-":
        pass
    else:
        key += 1
        arguments[key] = []
        line = x.split("--")
        for l in range(len(line)):
            arg = line[l].split(' ')
            arguments[key].append(arg)

for key in arguments:

    args = arguments[key]
    exp_name = args[28][-1].split('\n')[0]
    z_struct_size = int(args[11][1])
    stride_size = int(args[23][1])
    classif_layer_size = int(args[13][1])
    hidden_filters_1 = int(args[24][1])
    hidden_filters_2 = int(args[25][1])
    hidden_filters_3 = int(args[26][1])
    big_kernel_size = int(args[17][1])
    if args[27][1] == 'True':
        binary_z = True
    elif args[27][1] == 'False':
        binary_z = False
    if args[18][1] == 'True':
        two_conv_layer = True
    elif args[18][1] == 'False':
        two_conv_layer = False
    if args[19][1] == 'True':
        three_conv_layer = True
    elif args[19][1] == 'False':
        three_conv_layer = False
    if args[12][1] == 'True':
        add_classification_layer = True
    elif args[12][1] == 'False':
        add_classification_layer = False
    if args[20][1] == 'True':
        BK_in_first_layer = True
    elif args[20][1] == 'False':
        BK_in_first_layer = False
    if args[21][1] == 'True':
        BK_in_second_layer = True
    elif args[21][1] == 'False':
        BK_in_second_layer = False
    if args[22][1] == 'True':
        BK_in_third_layer = True
    elif args[22][1] == 'False':
        BK_in_third_layer = False

    net = Custom_CNN_BK(z_struct_size=z_struct_size,
                        big_kernel_size=big_kernel_size,
                        stride_size=stride_size,
                        classif_layer_size=classif_layer_size,
                        add_classification_layer=add_classification_layer,
                        hidden_filters_1=hidden_filters_1,
                        hidden_filters_2=hidden_filters_2,
                        hidden_filters_3=hidden_filters_3,
                        BK_in_first_layer=BK_in_first_layer,
                        two_conv_layer=two_conv_layer,
                        three_conv_layer=three_conv_layer,
                        BK_in_second_layer=BK_in_second_layer,
                        BK_in_third_layer=BK_in_third_layer,
                        Binary_z=binary_z)

    run_score(exp_name, net)

f.close()
