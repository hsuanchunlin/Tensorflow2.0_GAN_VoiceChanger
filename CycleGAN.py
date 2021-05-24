import os
#import tensorflow as tf
import tensorflow.compat.v2 as tf
from model_tf2 import *
#from model_tf2 import discriminator, generator_gatecnn
#from utils import l1_loss, l2_loss, cross_entropy_loss
from datetime import datetime

def l1_loss(y, y_hat):
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    return tf.reduce_mean(tf.abs(y - y_hat))


class CycleGAN(tf.keras.Model):
    def __init__(self, num_features=24,
                 lambda_cycle=10.0,
                 lambda_identity=5.0):
        super(CycleGAN, self).__init__()

        self.generation_A2B = generator_gatecnn()
        self.generation_B2A = generator_gatecnn()
        self.discrimination_A = discriminator()
        self.discrimination_B = discriminator()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.num_features = num_features

    def compile(
        self,
        #generator_optimizer,
        #discriminator_optimizer,
        generation_A2B_optimizer,
        generation_B2A_optimizer,
        discrimination_A_optimizer,
        discrimination_B_optimizer,
        gen_loss_fn,
        disc_loss_fn,

    ):
        super(CycleGAN, self).compile()
        self.generation_A2B_optimizer = generation_A2B_optimizer
        self.generation_B2A_optimizer = generation_B2A_optimizer
        self.discrimination_A_optimizer = discrimination_A_optimizer
        self.discrimination_B_optimizer = discrimination_B_optimizer
        #self.generator_optimizer = generator_optimizer
        #self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def save_model(self, filepath):
        tf.keras.models.save_model(self.generation_A2B, filepath+'_AB')
        tf.keras.models.save_model(self.generation_B2A, filepath+'_BA')

    #By adding tf.function, we can optimized the efficiency of training
    #@tf.function
    def train_step(self, batch):
        # x is Horse and y is zebra
        real_A, real_B = batch
        real_A_in = tf.expand_dims(real_A,0)
        real_B_in = tf.expand_dims(real_B,0)
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_B = self.generation_A2B(real_A_in, training=True)
            # Zebra to fake horse -> y2x
            fake_A = self.generation_B2A(real_B_in, training=True)

            # Cycle (Horse to fake zebra to fake horse): A -> B -> A
            cycled_A = self.generation_B2A(fake_B, training=True)
            # Cycle (Zebra to fake horse to fake zebra) B -> A -> B
            cycled_B = self.generation_A2B(fake_A, training=True)
            # Identity mapping (A->A => B2A(A))
            same_A = self.generation_B2A(real_A_in, training=True)
            same_B = self.generation_A2B(real_B_in, training=True)

            # Discriminator output

            disc_real_A = self.discrimination_A(real_A_in, training=True)
            disc_fake_A = self.discrimination_A(fake_A, training=True)

            disc_real_B = self.discrimination_B(real_B_in, training=True)
            disc_fake_B = self.discrimination_B(fake_B, training=True)

            # Generator adverserial loss
            #gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_A2B_loss = self.generator_loss_fn(tf.ones_like(disc_fake_B), disc_fake_B)
            #gen_F_loss = self.generator_loss_fn(disc_fake_x)
            gen_B2A_loss = self.generator_loss_fn(tf.ones_like(disc_fake_A), disc_fake_A)

            # Generator cycle loss
            #cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle

            cycle_loss_A2B = self.cycle_loss_fn(real_B_in, cycled_B)
            #cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle
            cycle_loss_B2A = self.cycle_loss_fn(real_A_in, cycled_A)

            cycle_loss = cycle_loss_A2B + cycle_loss_B2A

            # Generator identity loss
            id_loss_A2B = (
                self.identity_loss_fn(real_B_in, same_B)
            )
            id_loss_B2A = (
                self.identity_loss_fn(real_A_in, same_A)
            )
            identity_loss = id_loss_A2B + id_loss_B2A

            # Total generator loss
            total_loss_A2B = gen_A2B_loss + self.lambda_cycle * cycle_loss_A2B + self.lambda_identity * id_loss_A2B
            total_loss_B2A = gen_B2A_loss + self.lambda_cycle * cycle_loss_B2A + self.lambda_identity * id_loss_B2A
            generator_loss = gen_A2B_loss + gen_B2A_loss + self.lambda_cycle * cycle_loss + self.lambda_identity * identity_loss

            # Discriminator loss
            # Discriminator wants to classify real and fake correctly

            disc_A_real_loss = self.discriminator_loss_fn(tf.ones_like(disc_real_A), disc_real_A)
            disc_A_fake_loss = self.discriminator_loss_fn(tf.zeros_like(disc_fake_A), disc_fake_A)
            discriminator_loss_A = (disc_A_real_loss + disc_A_fake_loss)/2

            disc_B_real_loss = self.discriminator_loss_fn(tf.ones_like(disc_real_B), disc_real_B)
            disc_B_fake_loss = self.discriminator_loss_fn(tf.zeros_like(disc_fake_B), disc_fake_B)
            discriminator_loss_B = (disc_B_real_loss + disc_B_fake_loss)/2

            #discriminator_loss = discriminator_loss_A + discriminator_loss_B

        # trainable variable
        #gen_train_var = [v for v in self.generation_A2B.trainable_variables] + [q for q in self.generation_B2A.trainable_variables]
        #dis_train_var = [v for v in self.discrimination_A.trainable_variables] + [q for q in self.discrimination_B.trainable_variables]
        # Get the gradients for the generators
        grads_A2B = tape.gradient(generator_loss, self.generation_A2B.trainable_variables)
        grads_B2A = tape.gradient(generator_loss, self.generation_B2A.trainable_variables)

        #grads_gen = tape.gradient(generator_loss, gen_train_var)
        # Get the gradients for the discriminators
        disc_A_grads = tape.gradient(discriminator_loss_A, self.discrimination_A.trainable_variables)
        disc_B_grads = tape.gradient(discriminator_loss_B, self.discrimination_B.trainable_variables)
        #disc_grads = tape.gradient(discriminator_loss, dis_train_var)
        # Update the weights of the generators
        self.generation_A2B_optimizer.apply_gradients(
            zip(grads_A2B, self.generation_A2B.trainable_variables)
        )

        self.generation_B2A_optimizer.apply_gradients(
            zip(grads_B2A, self.generation_B2A.trainable_variables)
        )
        #self.generator_optimizer.apply_gradients(
        #    zip(grads_gen, gen_train_var)
        #)
        # Update the weights of the discriminators
        self.discrimination_A_optimizer.apply_gradients(
            zip(disc_A_grads, self.discrimination_A.trainable_variables)
        )
        self.discrimination_B_optimizer.apply_gradients(
            zip(disc_B_grads, self.discrimination_B.trainable_variables)
        )
        #self.discriminator_optimizer.apply_gradients(
        #    zip(disc_grads, dis_train_var)
        #)
        return {
            "G_loss": total_loss_A2B,
            "F_loss": total_loss_B2A,
            "D_X_loss": discriminator_loss_A,
            "D_Y_loss": discriminator_loss_B,
        }
