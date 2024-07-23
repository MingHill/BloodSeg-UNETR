2024-06-20 20:43:35,926 - INFO - {'BATCH_SIZE': 4096,
 'COMMENT': 'Run 08',
 'DATASOURCE': '/home/hmgillis/datasets/third-party/pytorch-datasets',
 'LEARNING_RATE': 0.00015,
 'LOG_DIR': 'logs/cifar/08/',
 'NUM_EPOCHS': 100,
 'RANDOM_SEED': 42,
 'SAVE_MODELS': 1,
 'SCHEDULER': 1,
 'WEIGHT_DECAY': 0.05}
2024-06-20 20:43:43,364 - INFO - Train -> 40000, 10 (samples, batches)
2024-06-20 20:43:43,364 - INFO - Valid -> 10000, 3 (samples, batches)
2024-06-20 20:43:43,364 - INFO - Test  -> 10000,  3  (samples, batches)
2024-06-20 20:43:43,460 - INFO - ViTMAEForPreTraining(
  (vit): ViTMAEModel(
    (embeddings): ViTMAEEmbeddings(
      (patch_embeddings): ViTMAEPatchEmbeddings(
        (projection): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
      )
    )
    (encoder): ViTMAEEncoder(
      (layer): ModuleList(
        (0-5): 6 x ViTMAELayer(
          (attention): ViTMAESdpaAttention(
            (attention): ViTMAESdpaSelfAttention(
              (query): Linear(in_features=128, out_features=128, bias=True)
              (key): Linear(in_features=128, out_features=128, bias=True)
              (value): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ViTMAESelfOutput(
              (dense): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTMAEIntermediate(
            (dense): Linear(in_features=128, out_features=256, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTMAEOutput(
            (dense): Linear(in_features=256, out_features=128, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (layernorm_after): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
  )
  (decoder): ViTMAEDecoder(
    (decoder_embed): Linear(in_features=128, out_features=64, bias=True)
    (decoder_layers): ModuleList(
      (0-1): 2 x ViTMAELayer(
        (attention): ViTMAESdpaAttention(
          (attention): ViTMAESdpaSelfAttention(
            (query): Linear(in_features=64, out_features=64, bias=True)
            (key): Linear(in_features=64, out_features=64, bias=True)
            (value): Linear(in_features=64, out_features=64, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (output): ViTMAESelfOutput(
            (dense): Linear(in_features=64, out_features=64, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (intermediate): ViTMAEIntermediate(
          (dense): Linear(in_features=64, out_features=128, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): ViTMAEOutput(
          (dense): Linear(in_features=128, out_features=64, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (layernorm_before): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (layernorm_after): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      )
    )
    (decoder_norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
    (decoder_pred): Linear(in_features=64, out_features=48, bias=True)
  )
)
2024-06-20 20:43:43,461 - INFO - ViTMAEConfig {
  "attention_probs_dropout_prob": 0.0,
  "decoder_hidden_size": 64,
  "decoder_intermediate_size": 128,
  "decoder_num_attention_heads": 4,
  "decoder_num_hidden_layers": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 128,
  "image_size": 48,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "layer_norm_eps": 1e-06,
  "mask_ratio": 0.75,
  "model_type": "vit_mae",
  "norm_pix_loss": false,
  "num_attention_heads": 4,
  "num_channels": 3,
  "num_hidden_layers": 6,
  "patch_size": 4,
  "qkv_bias": true,
  "transformers_version": "4.41.2"
}

2024-06-20 20:43:43,462 - INFO - Parameters: 907888
2024-06-20 20:43:43,462 - INFO - Trainable parameters: 880048

2024-06-20 20:43:43,462 - INFO - Scheduler: <torch.optim.lr_scheduler.OneCycleLR object at 0x7f1cf041ac50>
2024-06-20 20:43:46,867 - INFO - Epoch 1/100 - Train Loss: 0.2764, Valid Loss: 0.2770, 
2024-06-20 20:43:50,100 - INFO - Epoch 2/100 - Train Loss: 0.2660, Valid Loss: 0.2667, 
2024-06-20 20:43:53,339 - INFO - Epoch 3/100 - Train Loss: 0.2508, Valid Loss: 0.2514, 
2024-06-20 20:43:56,580 - INFO - Epoch 4/100 - Train Loss: 0.2296, Valid Loss: 0.2301, 
2024-06-20 20:43:59,821 - INFO - Epoch 5/100 - Train Loss: 0.2025, Valid Loss: 0.2028, 
2024-06-20 20:44:03,052 - INFO - Epoch 6/100 - Train Loss: 0.1719, Valid Loss: 0.1723, 
2024-06-20 20:44:06,289 - INFO - Epoch 7/100 - Train Loss: 0.1415, Valid Loss: 0.1418, 
2024-06-20 20:44:09,530 - INFO - Epoch 8/100 - Train Loss: 0.1149, Valid Loss: 0.1151, 
2024-06-20 20:44:12,771 - INFO - Epoch 9/100 - Train Loss: 0.0942, Valid Loss: 0.0945, 
2024-06-20 20:44:16,012 - INFO - Epoch 10/100 - Train Loss: 0.0802, Valid Loss: 0.0804, 
2024-06-20 20:44:19,254 - INFO - Epoch 11/100 - Train Loss: 0.0714, Valid Loss: 0.0717, 
2024-06-20 20:44:22,489 - INFO - Epoch 12/100 - Train Loss: 0.0663, Valid Loss: 0.0665, 
2024-06-20 20:44:25,730 - INFO - Epoch 13/100 - Train Loss: 0.0633, Valid Loss: 0.0635, 
2024-06-20 20:44:28,971 - INFO - Epoch 14/100 - Train Loss: 0.0616, Valid Loss: 0.0617, 
2024-06-20 20:44:32,212 - INFO - Epoch 15/100 - Train Loss: 0.0605, Valid Loss: 0.0606, 
2024-06-20 20:44:35,454 - INFO - Epoch 16/100 - Train Loss: 0.0597, Valid Loss: 0.0598, 
2024-06-20 20:44:38,689 - INFO - Epoch 17/100 - Train Loss: 0.0592, Valid Loss: 0.0593, 
2024-06-20 20:44:41,930 - INFO - Epoch 18/100 - Train Loss: 0.0587, Valid Loss: 0.0588, 
2024-06-20 20:44:45,172 - INFO - Epoch 19/100 - Train Loss: 0.0584, Valid Loss: 0.0585, 
2024-06-20 20:44:48,416 - INFO - Epoch 20/100 - Train Loss: 0.0581, Valid Loss: 0.0582, 
2024-06-20 20:44:51,661 - INFO - Epoch 21/100 - Train Loss: 0.0578, Valid Loss: 0.0579, 
2024-06-20 20:44:54,902 - INFO - Epoch 22/100 - Train Loss: 0.0576, Valid Loss: 0.0576, 
2024-06-20 20:44:58,149 - INFO - Epoch 23/100 - Train Loss: 0.0574, Valid Loss: 0.0575, 
2024-06-20 20:45:01,400 - INFO - Epoch 24/100 - Train Loss: 0.0572, Valid Loss: 0.0573, 
2024-06-20 20:45:04,652 - INFO - Epoch 25/100 - Train Loss: 0.0571, Valid Loss: 0.0572, 
2024-06-20 20:45:07,904 - INFO - Epoch 26/100 - Train Loss: 0.0570, Valid Loss: 0.0571, 
2024-06-20 20:45:11,156 - INFO - Epoch 27/100 - Train Loss: 0.0569, Valid Loss: 0.0569, 
2024-06-20 20:45:14,403 - INFO - Epoch 28/100 - Train Loss: 0.0568, Valid Loss: 0.0569, 
2024-06-20 20:45:17,655 - INFO - Epoch 29/100 - Train Loss: 0.0568, Valid Loss: 0.0569, 
2024-06-20 20:45:20,908 - INFO - Epoch 30/100 - Train Loss: 0.0566, Valid Loss: 0.0567, 
2024-06-20 20:45:24,160 - INFO - Epoch 31/100 - Train Loss: 0.0566, Valid Loss: 0.0567, 
2024-06-20 20:45:27,412 - INFO - Epoch 32/100 - Train Loss: 0.0566, Valid Loss: 0.0567, 
2024-06-20 20:45:30,658 - INFO - Epoch 33/100 - Train Loss: 0.0566, Valid Loss: 0.0567, 
2024-06-20 20:45:33,908 - INFO - Epoch 34/100 - Train Loss: 0.0566, Valid Loss: 0.0567, 
2024-06-20 20:45:37,160 - INFO - Epoch 35/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:45:40,411 - INFO - Epoch 36/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:45:43,662 - INFO - Epoch 37/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:45:46,902 - INFO - Epoch 38/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:45:50,151 - INFO - Epoch 39/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:45:53,403 - INFO - Epoch 40/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:45:56,656 - INFO - Epoch 41/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:45:59,909 - INFO - Epoch 42/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:46:03,163 - INFO - Epoch 43/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:46:06,411 - INFO - Epoch 44/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:46:09,664 - INFO - Epoch 45/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:46:12,918 - INFO - Epoch 46/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:46:16,171 - INFO - Epoch 47/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:46:19,424 - INFO - Epoch 48/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:46:22,671 - INFO - Epoch 49/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:46:25,923 - INFO - Epoch 50/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:46:29,176 - INFO - Epoch 51/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:46:32,429 - INFO - Epoch 52/100 - Train Loss: 0.0564, Valid Loss: 0.0564, 
2024-06-20 20:46:35,682 - INFO - Epoch 53/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:46:38,926 - INFO - Epoch 54/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:46:42,175 - INFO - Epoch 55/100 - Train Loss: 0.0563, Valid Loss: 0.0564, 
2024-06-20 20:46:45,428 - INFO - Epoch 56/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:46:48,682 - INFO - Epoch 57/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:46:51,935 - INFO - Epoch 58/100 - Train Loss: 0.0564, Valid Loss: 0.0564, 
2024-06-20 20:46:55,189 - INFO - Epoch 59/100 - Train Loss: 0.0565, Valid Loss: 0.0566, 
2024-06-20 20:46:58,438 - INFO - Epoch 60/100 - Train Loss: 0.0564, Valid Loss: 0.0564, 
2024-06-20 20:47:01,691 - INFO - Epoch 61/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:04,945 - INFO - Epoch 62/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:08,199 - INFO - Epoch 63/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:11,453 - INFO - Epoch 64/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:14,700 - INFO - Epoch 65/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:17,952 - INFO - Epoch 66/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:21,206 - INFO - Epoch 67/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:24,459 - INFO - Epoch 68/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:27,713 - INFO - Epoch 69/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:30,958 - INFO - Epoch 70/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:34,208 - INFO - Epoch 71/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:37,462 - INFO - Epoch 72/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:40,716 - INFO - Epoch 73/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:43,970 - INFO - Epoch 74/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:47,225 - INFO - Epoch 75/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:50,473 - INFO - Epoch 76/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:47:53,727 - INFO - Epoch 77/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:47:56,981 - INFO - Epoch 78/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:00,235 - INFO - Epoch 79/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:03,490 - INFO - Epoch 80/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:06,736 - INFO - Epoch 81/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:09,988 - INFO - Epoch 82/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:13,242 - INFO - Epoch 83/100 - Train Loss: 0.0564, Valid Loss: 0.0566, 
2024-06-20 20:48:16,497 - INFO - Epoch 84/100 - Train Loss: 0.0564, Valid Loss: 0.0566, 
2024-06-20 20:48:19,751 - INFO - Epoch 85/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:22,996 - INFO - Epoch 86/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:26,247 - INFO - Epoch 87/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:29,501 - INFO - Epoch 88/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:32,755 - INFO - Epoch 89/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:36,010 - INFO - Epoch 90/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:39,265 - INFO - Epoch 91/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:42,514 - INFO - Epoch 92/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:45,768 - INFO - Epoch 93/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:49,022 - INFO - Epoch 94/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:52,276 - INFO - Epoch 95/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:48:55,531 - INFO - Epoch 96/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:48:58,778 - INFO - Epoch 97/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:49:02,030 - INFO - Epoch 98/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:49:05,284 - INFO - Epoch 99/100 - Train Loss: 0.0564, Valid Loss: 0.0565, 
2024-06-20 20:49:08,540 - INFO - Epoch 100/100 - Train Loss: 0.0565, Valid Loss: 0.0565, 
2024-06-20 20:49:08,825 - INFO - Test Loss: 0.0600
2024-06-20 20:49:08,825 - INFO - Elapsed time -> 5.5485 minutes
