import os

import keras.backend as K
import numpy as np
from tqdm import tqdm

from utils.utils import show_result


def fit_one_epoch(
    G_model_A2B, G_model_B2A, D_model_A, D_model_B, Combine_model, G_model_A2B_body, G_model_B2A_body, D_model_A_body, D_model_B_body, \
    loss_history, epoch, epoch_step, gen, Epoch, save_period, save_dir, photo_save_step
):
    G_total_loss    = 0
    D_total_loss_A  = 0
    D_total_loss_B  = 0

    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images_A, images_B = batch[0], batch[1]
            batch_size  = np.shape(images_A)[0]
            y_real      = np.ones([batch_size, 1])
            y_fake      = np.zeros([batch_size, 1])

            #----------------------------------------------------#
            #   再训练生成器
            #   目的是让生成器生成的图像，被评价器认为是正确的
            #----------------------------------------------------#
            g_loss = Combine_model.train_on_batch([images_A, images_B],
                                                    [y_real, y_real,
                                                    images_A, images_B,
                                                    images_A, images_B])
            # ---------------------- #
            #  训练评价者
            # ---------------------- #
            # A到B的假图片，此时生成的是假橘子
            fake_B = G_model_A2B.predict(images_A)
            # B到A的假图片，此时生成的是假苹果
            fake_A = G_model_B2A.predict(images_B)
            # 判断真假图片，并以此进行训练
            dA_loss_real    = D_model_A.train_on_batch(images_A, y_real)
            dA_loss_fake    = D_model_A.train_on_batch(fake_A, y_fake)
            dA_loss         = 0.5 * np.add(dA_loss_real, dA_loss_fake)
            # 判断真假图片，并以此进行训练
            dB_loss_real    = D_model_B.train_on_batch(images_B, y_real)
            dB_loss_fake    = D_model_B.train_on_batch(fake_B, y_fake)
            dB_loss         = 0.5 * np.add(dB_loss_real, dB_loss_fake)


            G_total_loss    += g_loss[0]
            D_total_loss_A  += dA_loss
            D_total_loss_B  += dB_loss

            pbar.set_postfix(**{'G_loss'    : G_total_loss / (iteration + 1), 
                                'D_loss_A'  : D_total_loss_A / (iteration + 1), 
                                'D_loss_B'  : D_total_loss_B / (iteration + 1), 
                                'lr'        : K.get_value(Combine_model.optimizer.lr)},)
            pbar.update(1)

            if iteration % photo_save_step == 0:
                show_result(epoch + 1, G_model_A2B_body, G_model_B2A_body, images_A, images_B)

    G_total_loss    = G_total_loss / epoch_step
    D_total_loss_A  = D_total_loss_A / epoch_step
    D_total_loss_B  = D_total_loss_B / epoch_step

    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss A: %.4f || D Loss B: %.4f  ' % (G_total_loss, D_total_loss_A, D_total_loss_B))
    loss_history.append_loss(epoch + 1, G_total_loss = G_total_loss, D_total_loss_A = D_total_loss_A, D_total_loss_B = D_total_loss_B)

    #----------------------------#
    #   每若干个世代保存一次
    #----------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        G_model_A2B_body.save_weights(os.path.join(save_dir, 'G_model_A2B_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.h5'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))
        G_model_B2A_body.save_weights(os.path.join(save_dir, 'G_model_B2A_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.h5'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))
        D_model_A_body.save_weights(os.path.join(save_dir, 'D_model_A_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.h5'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))
        D_model_B_body.save_weights(os.path.join(save_dir, 'D_model_B_Epoch%d-GLoss%.4f-DALoss%.4f-DBLoss%.4f.h5'%(epoch + 1, G_total_loss, D_total_loss_A, D_total_loss_B)))

    if os.path.exists(os.path.join(save_dir, 'G_model_A2B_last_epoch_weights.h5')):
        os.remove(os.path.join(save_dir, 'G_model_A2B_last_epoch_weights.h5'))
    if os.path.exists(os.path.join(save_dir, 'G_model_B2A_last_epoch_weights.h5')):
        os.remove(os.path.join(save_dir, 'G_model_B2A_last_epoch_weights.h5'))
    if os.path.exists(os.path.join(save_dir, 'D_model_A_last_epoch_weights.h5')):
        os.remove(os.path.join(save_dir, 'D_model_A_last_epoch_weights.h5'))
    if os.path.exists(os.path.join(save_dir, 'D_model_B_last_epoch_weights.h5')):
        os.remove(os.path.join(save_dir, 'D_model_B_last_epoch_weights.h5'))
    G_model_A2B_body.save_weights(os.path.join(save_dir, "G_model_A2B_last_epoch_weights.h5"))
    G_model_B2A_body.save_weights(os.path.join(save_dir, "G_model_B2A_last_epoch_weights.h5"))
    D_model_A_body.save_weights(os.path.join(save_dir, "D_model_A_last_epoch_weights.h5"))
    D_model_B_body.save_weights(os.path.join(save_dir, "D_model_B_last_epoch_weights.h5"))