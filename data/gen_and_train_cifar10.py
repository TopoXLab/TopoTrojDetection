#!/usr/bin/env python3

import argparse
import glob
import logging.config
import multiprocessing
import os
import time
import sys
sys.path.insert(1, os.path.abspath("../"))
from numpy.random import RandomState

import torch
import trojai.modelgen.architecture_factory as tpm_af
import trojai.modelgen.architectures.cifar10_architectures as cfa

from networks import ResNet18
cfa.ResNet18=ResNet18 # Monkey Patch
import trojai.modelgen.config as tpmc
import trojai.modelgen.data_manager as tpm_tdm
import trojai.modelgen.model_generator as mg

import numpy as np

import cifar10
import trojai.datagen.merge_interface as td_merge
import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.datatype_xforms as tdd
import trojai.datagen.experiment as tde
import trojai.datagen.insert_merges as tdi
import trojai.datagen.config as tdc
import trojai.datagen.image_triggers as tdt
import trojai.datagen.xform_merge_pipeline as tdx
import trojai.datagen.instagram_xforms as tinstx

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CIFAR10 Data & Model Generator and Experiment Iterator')
    # args related to data generation
    parser.add_argument('--experiment_path', type=str, help='Top Folder of output')
    parser.add_argument('--data_folder', type=str, help='Path to folder containing generated example images')
    # args related to model generation
    parser.add_argument('--log', type=str, help='Log File')
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--models_output', type=str, help='Folder in which to save models')
    parser.add_argument('--tensorboard_dir', type=str, default='/tmp/tensorboard', help='Folder for logging tensorboard')
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--gpu_index', default='0', help='Index of GPUs to be used')
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_val_split', help='Amount of train data to use for validation', default=0.1, type=float)
    parser.add_argument('--num_models', type=int, help='Number of models to be generated')
    parser.add_argument('--troj_frac', type=float, help='Trojan fraction')
    parser.add_argument('--target_class', type=str, help='Target class to be flipped to')
    parser.add_argument('--network', type=str, help='Architecture to be used', default='resnet18', choices={'resnet18', 'densenet121'})
    a = parser.parse_args()

    logger = logging.getLogger(__name__)

    class DummyMerge(td_merge.Merge):
        def do(self, obj1, obj2, random_state_obj):
            pass

    class CIFAR10ArchFactory(tpm_af.ArchitectureFactory):
        def new_architecture(self):
            # return cfa.AlexNet()
            if a.network=='densenet121':
                return cfa.densenet_cifar()
            if a.network=='resnet18':
                return cfa.ResNet18(in_channel=3, num_classes=10)

    def img_transform(x):
        # xform data to conform to PyTorch
        x = x.permute(2, 0, 1)
        return x

    # setup logger
    handlers = []
    if a.log is not None:
        log_fname = a.log
        handlers.append('file')
    else:
        log_fname = '/dev/null'
    if a.console is not None:
        handlers.append('console')

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'basic': {
                'format': '%(message)s',
            },
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_fname,
                'maxBytes': 1 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'detailed',
                'level': 'INFO',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'basic',
                'level': 'INFO',
            }
        },
        'loggers': {
            'trojai': {
                'handlers': handlers,
            },
        },
        'root': {
            'level': 'INFO',
        },
    })

    # setup data generation
    # Setup the files based on user inputs
    data_folder = os.path.abspath(a.data_folder)
    toplevel_folder = a.experiment_path

    # check if the data_folder has the cifar10 data, if not download it
    data_folder = cifar10.download_and_extract(data_folder)

    train_output_csv_file = 'train_cifar10.csv'
    test_output_csv_file = 'test_cifar10.csv'

    MASTER_SEED = 1234
    master_random_state_object = RandomState(MASTER_SEED)
    start_state = master_random_state_object.get_state()

    args_trigger_frac=a.troj_frac
    args_target_class=[int(x) for x in a.target_class.split(",")]
    # define a configuration which triggers data by applying the Gotham Instagram Filter
    datagen_per_class_trigger_frac = args_trigger_frac
    trigger_position=np.repeat(np.asarray([[int(np.random.choice([28, 0])),
                                            int(np.random.choice([28, 0]))]]), 3, 0)
    reverselambda_trigger_cfg = \
        tdc.XFormMergePipelineConfig(
            # setup the list of possible triggers that will be inserted into the CIFAR10 data.
            trigger_list=[tdt.ReverseLambdaPattern(3, 3, 3, 255)],
            # tell the trigger inserter the probability of sampling each type of trigger specified in the trigger
            # list.  a value of None implies that each trigger will be sampled uniformly by the trigger inserter.
            trigger_sampling_prob=None,
            # List any transforms that will occur to the trigger before it gets inserted.  In this case, we do none.
            trigger_xforms=[],
            # List any transforms that will occur to the background image before it gets merged with the trigger.
            # trigger_bg_xforms=[tinstx.GothamFilterXForm()],
            trigger_bg_xforms=[tdd.ToTensorXForm()],
            # List how we merge the trigger and the background.  Because we don't insert a point trigger,
            # the merge is just a no-op
            # trigger_bg_merge=DummyMerge(),
            # A list of any transformations that we should perform after merging the trigger and the background.
            trigger_bg_merge=tdi.InsertAtLocation(trigger_position),
            trigger_bg_merge_xforms=[],
            # Denotes how we merge the trigger with the background.
            merge_type='insert',
            # Specify that all the clean data will be modified.  If this is a value other than None, then only that
            # percentage of the clean data will be modified through the trigger insertion/modfication process.
            # per_class_trigger_frac=datagen_per_class_trigger_frac,
            # Specify which classes will be triggered
            per_class_trigger_frac=0.01 if args_trigger_frac==0 else args_trigger_frac, # This is hacky since set per_class_trigger_frac=0 will raise error
            triggered_classes=args_target_class
        )

    ############# Create the data ############
    # create the clean data
    clean_dataset_rootdir = os.path.join(toplevel_folder, 'cifar10_clean')
    master_random_state_object.set_state(start_state)
    cifar10.create_clean_dataset(data_folder, clean_dataset_rootdir, train_output_csv_file, test_output_csv_file,
                                 'cifar10_train_', 'cifar10_test_', [], master_random_state_object)
    # create a triggered version of the train data according to the configuration above
    mod_dataset_rootdir = 'cifar10_reverselambda_trigger'
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, train_output_csv_file,
                                   toplevel_folder, mod_dataset_rootdir,
                                   reverselambda_trigger_cfg, 'insert', master_random_state_object)
    # create a triggered version of the test data according to the configuration above
    master_random_state_object.set_state(start_state)
    tdx.modify_clean_image_dataset(clean_dataset_rootdir, test_output_csv_file,
                                   toplevel_folder, mod_dataset_rootdir,
                                   reverselambda_trigger_cfg, 'insert', master_random_state_object)

    ############# Create experiments from the data ############
    # Create a clean data experiment, which is just the original CIFAR10 experiment where clean data is used for
    # training and testing the model
    trigger_frac = 0.0
    trigger_behavior = tdb.WrappedAdd(1, 10)
    e = tde.ClassicExperiment(toplevel_folder, trigger_behavior)
    train_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean', 'train_cifar10.csv'),
                                   clean_dataset_rootdir,
                                   mod_filename_filter='*train*',
                                   split_clean_trigger=False,
                                   trigger_frac=trigger_frac,
                                   triggered_classes=args_target_class)
    train_df.to_csv(os.path.join(toplevel_folder, 'cifar10_clean_experiment_train.csv'), index=None)
    test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean', 'test_cifar10.csv'),
                                                           clean_dataset_rootdir,
                                                           mod_filename_filter='*test*',
                                                           split_clean_trigger=True,
                                                           trigger_frac=trigger_frac,
                                                           triggered_classes=args_target_class)
    test_clean_df.to_csv(os.path.join(toplevel_folder, 'cifar10_clean_experiment_test_clean.csv'), index=None)
    test_triggered_df.to_csv(os.path.join(toplevel_folder, 'cifar10_clean_experiment_test_triggered.csv'), index=None)

    # Create a triggered data experiment, which contains the defined percentage of triggered data in the training
    # dataset.  The remaining training data is clean data.  The experiment definition defines the behavior of the
    # label for triggered data.  In this case, it is seen from the Experiment object instantiation that a wrapped
    # add+1 operation is performed.
    # In the code below, we create several experiments with varying levels of poisoned data to allow for
    # experimentation.
    trigger_fracs = [args_trigger_frac]
    for trigger_frac in trigger_fracs:
        train_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean', 'train_cifar10.csv'),
                                       os.path.join(toplevel_folder, mod_dataset_rootdir),
                                       mod_filename_filter='*train*',
                                       split_clean_trigger=False,
                                       trigger_frac=trigger_frac,
                                       triggered_classes=args_target_class)
        train_df.to_csv(os.path.join(toplevel_folder, 'cifar10_reverselambda_'+str(trigger_frac)+'_experiment_train.csv'), index=None)
        test_clean_df, test_triggered_df = e.create_experiment(os.path.join(toplevel_folder, 'cifar10_clean', 'test_cifar10.csv'),
                                                               os.path.join(toplevel_folder, mod_dataset_rootdir),
                                                               mod_filename_filter='*test*',
                                                               split_clean_trigger=True,
                                                               trigger_frac=trigger_frac,
                                                               triggered_classes=args_target_class)
        test_clean_df.to_csv(os.path.join(toplevel_folder, 'cifar10_reverselambda_'+str(trigger_frac)+'_experiment_test_clean.csv'), index=None)
        test_triggered_df.to_csv(os.path.join(toplevel_folder, 'cifar10_reverselambda_'+str(trigger_frac)+'_experiment_test_triggered.csv'), index=None)

    # get all available experiments from the experiment root directory
    my_experiment_path = a.experiment_path
    flist = glob.glob(os.path.join(my_experiment_path, '*.csv'))
    experiment_name_list = list(set([os.path.basename(x.split('_experiment_')[0]) for x in flist]))
    experiment_name_list = [x for x in experiment_name_list if '0.' in x]
    experiment_name_list.sort()
    experiment_list = []
    for experiment_name in experiment_name_list:
        train_file = os.path.join(toplevel_folder, experiment_name + '_experiment_train.csv')
        clean_test_file = os.path.join(toplevel_folder, experiment_name + '_experiment_test_clean.csv')
        triggered_test_file = os.path.join(toplevel_folder, experiment_name + '_experiment_test_triggered.csv')

        if not (os.path.exists(train_file) and os.path.exists(clean_test_file) and os.path.exists(triggered_test_file)):
            warning_msg = 'Skipping experiment=' + experiment_name + ' because all the required files do not exist!'
            logger.warning(warning_msg)

        experiment_cfg = dict()
        experiment_cfg['train_file'] = train_file
        experiment_cfg['clean_test_file'] = clean_test_file
        experiment_cfg['triggered_test_file'] = triggered_test_file
        experiment_cfg['model_save_dir'] = experiment_name
        experiment_cfg['stats_save_dir'] = experiment_name
        experiment_cfg['experiment_path'] = my_experiment_path
        experiment_cfg['name'] = experiment_name
        experiment_list.append(experiment_cfg)

    model_save_root_dir = a.models_output
    stats_save_root_dir = a.models_output

    arch = CIFAR10ArchFactory()
    logger.warning("Using architecture:" + str(arch))
    logger.warning("Ensure that architecture matches dataset!")

    num_avail_cpus = multiprocessing.cpu_count()
    num_cpus_to_use = int(.8 * num_avail_cpus)

    modelgen_cfgs = []
    for i in range(len(experiment_list)):
        experiment_cfg = experiment_list[i]

        experiment_name = experiment_name_list[i]
        logger.debug(experiment_name)

        data_obj = tpm_tdm.DataManager(my_experiment_path,
                                       experiment_cfg['train_file'],
                                       experiment_cfg['clean_test_file'],
                                       triggered_test_file=experiment_cfg['triggered_test_file'],
                                       train_data_transform=img_transform,
                                       test_data_transform=img_transform,
                                       shuffle_train=True,
                                       train_dataloader_kwargs={'num_workers': num_cpus_to_use})

        model_save_dir = os.path.join(model_save_root_dir, experiment_cfg['model_save_dir'])
        stats_save_dir = os.path.join(model_save_root_dir, experiment_cfg['stats_save_dir'])
        num_models = 1

        device = torch.device('cuda' if torch.cuda.is_available() and a.gpu else 'cpu')

        default_nbpvdm = None if device.type == 'cpu' else 500

        early_stopping_argin = tpmc.EarlyStoppingConfig(num_epochs=8) if a.early_stopping else None
        training_params = tpmc.TrainingConfig(device=device,
                                              epochs=a.num_epochs,
                                              batch_size=32,
                                              lr=0.001,
                                              optim='adam',
                                              objective='cross_entropy_loss',
                                              early_stopping=early_stopping_argin,
                                              train_val_split=a.train_val_split)
        reporting_params = tpmc.ReportingConfig(num_batches_per_logmsg=500,
                                                num_epochs_per_metric=1,
                                                num_batches_per_metrics=default_nbpvdm,
                                                tensorboard_output_dir=a.tensorboard_dir,
                                                experiment_name=experiment_cfg['name'])
        optimizer_cfg = tpmc.DefaultOptimizerConfig(training_params, reporting_params)

        cfg = tpmc.ModelGeneratorConfig(arch, data_obj, model_save_dir, stats_save_dir, num_models,
                                        optimizer=optimizer_cfg,
                                        experiment_cfg=experiment_cfg,
                                        parallel=True)
        # may also provide lists of run_ids or filenames are arguments to ModelGeneratorConfig to have more control
        # of saved model file names; see RunnerConfig and ModelGeneratorConfig for more information
        modelgen_cfgs.append(cfg)

    model_generator = mg.ModelGenerator(modelgen_cfgs)
    start = time.time()
    model_generator.run()
    print("\nTime to run: ", (time.time() - start) / 60 / 60, 'hours')
