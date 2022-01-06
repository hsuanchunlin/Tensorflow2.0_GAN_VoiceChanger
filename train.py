import tensorflow as tf
import os
import numpy as np
import argparse
import time
import librosa
from preprocess import *
import soundfile as sf
import model
from CycleGAN import *
from tqdm import tqdm
import datetime




num_epochs = 300
dataset_A = np.load('./data/preprocessed/George.npy')
dataset_B = np.load('./data/preprocessed/Joanne.npy')

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))


model = CycleGAN(num_features = 24,
lambda_cycle=10.0,
lambda_identity=5.0)

# Restore the weights
#model.load_weights('./tf2/my_checkpoint')

#Setting learning rate decay for better converge
lr_schedule_gen = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.0002,
    decay_steps = 20000,
    end_learning_rate=1e-6,
    power=1.0
)
lr_schedule_dis = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.01,
    decay_steps = 20000,
    end_learning_rate=1e-6,
    power=1.0
)
lr_exponential_gen = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.0002,
    decay_steps = 10000,
    decay_rate = 0.96,
    staircase=False,
    name=None
)
lr_exponential_dis = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.0001,
    decay_steps = 10000,
    decay_rate = 0.96,
    staircase=False,
    name=None
)

# Compile the model
model.compile(
    generation_A2B_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen, beta_1=0.5),
    generation_B2A_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen, beta_1=0.5),
    discrimination_A_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_exponential_dis, beta_1=0.5),
    discrimination_B_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_exponential_dis, beta_1=0.5),
    #generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_exponential_gen, beta_1=0.5),
    #discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_exponential_dis, beta_1=0.5),
    gen_loss_fn=l2_loss,
    disc_loss_fn=l2_loss,
)
#Training
    #"G_loss": total_loss_A2B,
    #"F_loss": total_loss_B2A,
    #"D_X_loss": disc_A_loss,
    #"D_Y_loss": disc_B_loss,


# For validation
data = np.load('model_Joanne-split/logf0s_normalization.npz')
data2 = np.load('model_Joanne-split/mcep_normalization.npz')


class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, dataA, dataB):
        self.log_f0s_mean_A = dataA['mean_A']
        self.log_f0s_mean_B = dataA['mean_B']
        self.log_f0s_std_A = dataA['std_A']
        self.log_f0s_std_B = dataA['std_B']
        self.coded_sps_A_mean = dataB['mean_A']
        self.coded_sps_B_mean = dataB['mean_B']
        self.coded_sps_A_std = dataB['std_A']
        self.coded_sps_B_std = dataB['std_B']
        self.sampling_rate = 16000
        self.num_mcep = 24
        self.frame_period = 5.0
        self.n_frames = 128

    def on_epoch_end(self, epoch, logs=None):
        print('Starting evaluation step..........')
        wav, _ = librosa.load('data/voiceA-George/02.wav', sr = self.sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = self.sampling_rate, frame_period = self.frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = self.sampling_rate, frame_period = self.frame_period)
        f0_converted = pitch_conversion(f0 = f0, mean_log_src = self.log_f0s_mean_A, std_log_src = self.log_f0s_std_A, mean_log_target = self.log_f0s_mean_B, std_log_target = self.log_f0s_std_B)
        coded_sp = world_encode_spectral_envelop(sp = self.sampling_rate, fs = self.sampling_rate, dim = self.num_mcep)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = (coded_sp_transposed - self.coded_sps_A_mean) / self.coded_sps_A_std
        coded_sp_converted_norm = model.generation_A2B(tf.expand_dims(coded_sp_norm,0))
        coded_sp_converted_norm = tf.squeeze(coded_sp_converted_norm)
        coded_sp_converted = coded_sp_converted_norm * self.coded_sps_B_std + self.coded_sps_B_mean
        coded_sp_converted = tf.transpose(coded_sp_converted)
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        coded_sp_converted = coded_sp_converted.astype('double')
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = self.sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = self.sampling_rate, frame_period = self.frame_period)
        file_path = ("data/output/tf2/ver_{epoch}.wav".format(epoch=epoch + 1))
        sf.write(file_path, wav_transformed, self.sampling_rate)
#wav_gen = GANMonitor(data,data2)

dataA = tf.data.Dataset.from_tensor_slices(dataset_A)
dataB = tf.data.Dataset.from_tensor_slices(dataset_B)

# Create a callback that saves the model's weights
log_dir = "logs/fit/J/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./tf2/J/Joanne',
                                                 save_weights_only=True,
                                                 verbose=1)
model.load_weights('./tf2/J/Joanne')
model.fit(tf.data.Dataset.zip((dataA, dataB)),
    callbacks=[tensorboard_callback],
    epochs = num_epochs)
print('Saving weights and check points')
model.save_weights('./tf2/J/Joanne')
model.save_model('./models/Joanne')
