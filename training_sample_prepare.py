import tensorflow as tf
import os
import numpy as np
import argparse
import time
import librosa
from preprocess import *



random_seed = 0
np.random.seed(random_seed)

num_epochs = 20
mini_batch_size = 1 # mini_batch_size = 1 is better
generator_learning_rate = 0.0002
generator_learning_rate_decay = generator_learning_rate / 200000
discriminator_learning_rate = 0.0001
discriminator_learning_rate_decay = discriminator_learning_rate / 200000
sampling_rate = 16000
#sampling_rate = 64000
num_mcep = 24
frame_period = 5.0
n_frames = 128
lambda_cycle = 10
lambda_identity = 5
print('Preprocessing Data...')
sr = sampling_rate

start_time = time.time()
train_A_dir = 'data/voiceA'
train_B_dir = 'data/voiceB'
#wavs_A = [wav]
wavs_A = load_wavs(wav_dir = train_A_dir, sr = sampling_rate)
wavs_B = load_wavs(wav_dir = train_B_dir, sr = sampling_rate)

f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)

log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

print('Log Pitch A')
print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
print('Log Pitch B')
print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))


coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)

coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
print("Input data fixed.")
coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)

model_dir = 'model_Joanne_nomatch'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)


dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

np.save('./data/preprocessed/nomat_George',dataset_A)
np.save('./data/preprocessed/nomat_Joanne',dataset_B)
