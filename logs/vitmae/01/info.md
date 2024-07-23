2024-06-22 18:24:48,567 - INFO - {'ATTENTION_PROBS_DROPOUT_PROB': 0.0,
 'BATCH_SIZE': 128,
 'BETA_1': 0.9,
 'BETA_2': 0.999,
 'COMMENT': 'Run 01',
 'DECODER_HIDDEN_SIZE': 64,
 'DECODER_INTERMEDIATE_SIZE': 128,
 'DECODER_NUM_ATTENTION_HEADS': 4,
 'DECODER_NUM_HIDDEN_LAYERS': 2,
 'DIV_FACTOR': 100,
 'FINAL_DIV_FACTOR': 1.0,
 'HIDDEN_DROPOUT_PROB': 0.0,
 'HIDDEN_SIZE': 128,
 'IMAGE_SIZE': 64,
 'INTERMEDIATE_SIZE': 256,
 'LEARNING_RATE': 0.00015,
 'LOG_DIR': 'logs/vitmae/01/',
 'MASK_RATIO': 0.75,
 'NUM_ATTENTION_HEADS': 4,
 'NUM_CHANNELS': 16,
 'NUM_EPOCHS': 5,
 'NUM_HIDDEN_LAYERS': 6,
 'PATCH_SIZE': 4,
 'PCT_START': 0.1,
 'RANDOM_SEED': 42,
 'SAVE_MODELS': 1,
 'SCHEDULER': 0,
 'TEST_DATASET': '$SLURM_TMPDIR/datasets/testing_dataset.npy',
 'TRAIN_DATASET': '$SLURM_TMPDIR/datasets/training_dataset.npy',
 'VALID_DATASET': '$SLURM_TMPDIR/datasets/validation_dataset.npy',
 'WEIGHT_DECAY': 0.0}
2024-06-22 18:24:58,618 - INFO - Train -> 294912, 2304 (samples, batches)
2024-06-22 18:24:58,618 - INFO - Valid -> 36864, 288 (samples, batches)
2024-06-22 18:24:58,618 - INFO - Test  -> 36864,  288  (samples, batches)
2024-06-22 18:24:58,939 - INFO - ViTMAEForPreTraining(
  (vit): ViTMAEModel(
    (embeddings): ViTMAEEmbeddings(
      (patch_embeddings): ViTMAEPatchEmbeddings(
        (projection): Conv2d(16, 128, kernel_size=(4, 4), stride=(4, 4))
      )
    )
    (encoder): ViTMAEEncoder(
      (layer): ModuleList(
        (0-5): 6 x ViTMAELayer(
          (attention): ViTMAEAttention(
            (attention): ViTMAESelfAttention(
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
          (layernorm_before): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
  )
  (decoder): ViTMAEDecoder(
    (decoder_embed): Linear(in_features=128, out_features=64, bias=True)
    (decoder_layers): ModuleList(
      (0-1): 2 x ViTMAELayer(
        (attention): ViTMAEAttention(
          (attention): ViTMAESelfAttention(
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
        (layernorm_before): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
        (layernorm_after): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
      )
    )
    (decoder_norm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
    (decoder_pred): Linear(in_features=64, out_features=256, bias=True)
  )
)
2024-06-22 18:24:58,941 - INFO - ViTMAEConfig {
  "attention_probs_dropout_prob": 0.0,
  "decoder_hidden_size": 64,
  "decoder_intermediate_size": 128,
  "decoder_num_attention_heads": 4,
  "decoder_num_hidden_layers": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 128,
  "image_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "layer_norm_eps": 1e-12,
  "mask_ratio": 0.75,
  "model_type": "vit_mae",
  "norm_pix_loss": false,
  "num_attention_heads": 4,
  "num_channels": 16,
  "num_hidden_layers": 6,
  "patch_size": 4,
  "qkv_bias": true,
  "transformers_version": "4.40.2"
}

2024-06-22 18:24:58,941 - INFO - Parameters: 969536
2024-06-22 18:24:58,941 - INFO - Trainable parameters: 920192

2024-06-22 18:24:58,942 - INFO - Scheduler: None
2024-06-22 18:25:00,895 - INFO - Epoch 1/5, Batch 1/144 Batch Loss: 0.4463
2024-06-22 18:25:02,669 - INFO - Epoch 1/5, Batch 2/144 Batch Loss: 0.4409
2024-06-22 18:25:04,400 - INFO - Epoch 1/5, Batch 3/144 Batch Loss: 0.4254
2024-06-22 18:25:05,950 - INFO - Epoch 1/5, Batch 4/144 Batch Loss: 0.3985
2024-06-22 18:25:07,476 - INFO - Epoch 1/5, Batch 5/144 Batch Loss: 0.4213
2024-06-22 18:25:09,004 - INFO - Epoch 1/5, Batch 6/144 Batch Loss: 0.3965
2024-06-22 18:25:10,544 - INFO - Epoch 1/5, Batch 7/144 Batch Loss: 0.4267
2024-06-22 18:25:12,234 - INFO - Epoch 1/5, Batch 8/144 Batch Loss: 0.4172
2024-06-22 18:25:13,777 - INFO - Epoch 1/5, Batch 9/144 Batch Loss: 0.3813
2024-06-22 18:25:15,358 - INFO - Epoch 1/5, Batch 10/144 Batch Loss: 0.3790
2024-06-22 18:25:16,904 - INFO - Epoch 1/5, Batch 11/144 Batch Loss: 0.3559
2024-06-22 18:25:18,477 - INFO - Epoch 1/5, Batch 12/144 Batch Loss: 0.3837
2024-06-22 18:25:20,047 - INFO - Epoch 1/5, Batch 13/144 Batch Loss: 0.3550
2024-06-22 18:25:21,618 - INFO - Epoch 1/5, Batch 14/144 Batch Loss: 0.3364
2024-06-22 18:25:23,198 - INFO - Epoch 1/5, Batch 15/144 Batch Loss: 0.3524
2024-06-22 18:25:24,735 - INFO - Epoch 1/5, Batch 16/144 Batch Loss: 0.3127
2024-06-22 18:25:26,258 - INFO - Epoch 1/5, Batch 17/144 Batch Loss: 0.3202
2024-06-22 18:25:27,776 - INFO - Epoch 1/5, Batch 18/144 Batch Loss: 0.2987
2024-06-22 18:25:29,299 - INFO - Epoch 1/5, Batch 19/144 Batch Loss: 0.2970
2024-06-22 18:25:30,823 - INFO - Epoch 1/5, Batch 20/144 Batch Loss: 0.2926
2024-06-22 18:25:32,340 - INFO - Epoch 1/5, Batch 21/144 Batch Loss: 0.2754
2024-06-22 18:25:33,897 - INFO - Epoch 1/5, Batch 22/144 Batch Loss: 0.2732
2024-06-22 18:25:35,417 - INFO - Epoch 1/5, Batch 23/144 Batch Loss: 0.2544
2024-06-22 18:25:36,939 - INFO - Epoch 1/5, Batch 24/144 Batch Loss: 0.2577
2024-06-22 18:25:38,460 - INFO - Epoch 1/5, Batch 25/144 Batch Loss: 0.2585
2024-06-22 18:25:39,981 - INFO - Epoch 1/5, Batch 26/144 Batch Loss: 0.2672
2024-06-22 18:25:41,524 - INFO - Epoch 1/5, Batch 27/144 Batch Loss: 0.2247
2024-06-22 18:25:43,044 - INFO - Epoch 1/5, Batch 28/144 Batch Loss: 0.2292
2024-06-22 18:25:44,568 - INFO - Epoch 1/5, Batch 29/144 Batch Loss: 0.2284
2024-06-22 18:25:46,124 - INFO - Epoch 1/5, Batch 30/144 Batch Loss: 0.2494
2024-06-22 18:25:47,652 - INFO - Epoch 1/5, Batch 31/144 Batch Loss: 0.2193
2024-06-22 18:25:49,175 - INFO - Epoch 1/5, Batch 32/144 Batch Loss: 0.2146
2024-06-22 18:25:50,697 - INFO - Epoch 1/5, Batch 33/144 Batch Loss: 0.2103
2024-06-22 18:25:52,215 - INFO - Epoch 1/5, Batch 34/144 Batch Loss: 0.2014
2024-06-22 18:25:53,735 - INFO - Epoch 1/5, Batch 35/144 Batch Loss: 0.1887
2024-06-22 18:25:55,257 - INFO - Epoch 1/5, Batch 36/144 Batch Loss: 0.1973
2024-06-22 18:25:56,780 - INFO - Epoch 1/5, Batch 37/144 Batch Loss: 0.1914
2024-06-22 18:25:58,307 - INFO - Epoch 1/5, Batch 38/144 Batch Loss: 0.1669
2024-06-22 18:25:59,862 - INFO - Epoch 1/5, Batch 39/144 Batch Loss: 0.1784
2024-06-22 18:26:01,388 - INFO - Epoch 1/5, Batch 40/144 Batch Loss: 0.1843
2024-06-22 18:26:02,917 - INFO - Epoch 1/5, Batch 41/144 Batch Loss: 0.1708
2024-06-22 18:26:04,442 - INFO - Epoch 1/5, Batch 42/144 Batch Loss: 0.1710
2024-06-22 18:26:05,971 - INFO - Epoch 1/5, Batch 43/144 Batch Loss: 0.1606
2024-06-22 18:26:07,494 - INFO - Epoch 1/5, Batch 44/144 Batch Loss: 0.1583
2024-06-22 18:26:09,061 - INFO - Epoch 1/5, Batch 45/144 Batch Loss: 0.1423
2024-06-22 18:26:10,582 - INFO - Epoch 1/5, Batch 46/144 Batch Loss: 0.1469
2024-06-22 18:26:12,112 - INFO - Epoch 1/5, Batch 47/144 Batch Loss: 0.1455
2024-06-22 18:26:13,632 - INFO - Epoch 1/5, Batch 48/144 Batch Loss: 0.1379
2024-06-22 18:26:15,153 - INFO - Epoch 1/5, Batch 49/144 Batch Loss: 0.1341
2024-06-22 18:26:16,672 - INFO - Epoch 1/5, Batch 50/144 Batch Loss: 0.1320
2024-06-22 18:26:18,220 - INFO - Epoch 1/5, Batch 51/144 Batch Loss: 0.1366
2024-06-22 18:26:19,747 - INFO - Epoch 1/5, Batch 52/144 Batch Loss: 0.1275
2024-06-22 18:26:21,266 - INFO - Epoch 1/5, Batch 53/144 Batch Loss: 0.1282
2024-06-22 18:26:22,785 - INFO - Epoch 1/5, Batch 54/144 Batch Loss: 0.1215
2024-06-22 18:26:24,303 - INFO - Epoch 1/5, Batch 55/144 Batch Loss: 0.1214
2024-06-22 18:26:25,821 - INFO - Epoch 1/5, Batch 56/144 Batch Loss: 0.1140
2024-06-22 18:26:27,344 - INFO - Epoch 1/5, Batch 57/144 Batch Loss: 0.1167
2024-06-22 18:26:28,863 - INFO - Epoch 1/5, Batch 58/144 Batch Loss: 0.1098
2024-06-22 18:26:30,409 - INFO - Epoch 1/5, Batch 59/144 Batch Loss: 0.1111
2024-06-22 18:26:31,941 - INFO - Epoch 1/5, Batch 60/144 Batch Loss: 0.1104
2024-06-22 18:26:33,461 - INFO - Epoch 1/5, Batch 61/144 Batch Loss: 0.1057
2024-06-22 18:26:34,981 - INFO - Epoch 1/5, Batch 62/144 Batch Loss: 0.1068
2024-06-22 18:26:36,502 - INFO - Epoch 1/5, Batch 63/144 Batch Loss: 0.1049
2024-06-22 18:26:38,025 - INFO - Epoch 1/5, Batch 64/144 Batch Loss: 0.1042
2024-06-22 18:26:39,544 - INFO - Epoch 1/5, Batch 65/144 Batch Loss: 0.1034
2024-06-22 18:26:41,066 - INFO - Epoch 1/5, Batch 66/144 Batch Loss: 0.1003
2024-06-22 18:26:42,588 - INFO - Epoch 1/5, Batch 67/144 Batch Loss: 0.1023
2024-06-22 18:26:44,112 - INFO - Epoch 1/5, Batch 68/144 Batch Loss: 0.0978
2024-06-22 18:26:45,665 - INFO - Epoch 1/5, Batch 69/144 Batch Loss: 0.0983
2024-06-22 18:26:47,183 - INFO - Epoch 1/5, Batch 70/144 Batch Loss: 0.0968
2024-06-22 18:26:48,704 - INFO - Epoch 1/5, Batch 71/144 Batch Loss: 0.0938
2024-06-22 18:26:50,231 - INFO - Epoch 1/5, Batch 72/144 Batch Loss: 0.0982
2024-06-22 18:26:51,749 - INFO - Epoch 1/5, Batch 73/144 Batch Loss: 0.0967
2024-06-22 18:26:53,283 - INFO - Epoch 1/5, Batch 74/144 Batch Loss: 0.0927
2024-06-22 18:26:54,807 - INFO - Epoch 1/5, Batch 75/144 Batch Loss: 0.0932
2024-06-22 18:26:56,323 - INFO - Epoch 1/5, Batch 76/144 Batch Loss: 0.0933
2024-06-22 18:26:57,856 - INFO - Epoch 1/5, Batch 77/144 Batch Loss: 0.0900
2024-06-22 18:26:59,378 - INFO - Epoch 1/5, Batch 78/144 Batch Loss: 0.0912
2024-06-22 18:27:00,894 - INFO - Epoch 1/5, Batch 79/144 Batch Loss: 0.0891
2024-06-22 18:27:02,442 - INFO - Epoch 1/5, Batch 80/144 Batch Loss: 0.0885
2024-06-22 18:27:03,963 - INFO - Epoch 1/5, Batch 81/144 Batch Loss: 0.0920
2024-06-22 18:27:05,480 - INFO - Epoch 1/5, Batch 82/144 Batch Loss: 0.0937
2024-06-22 18:27:07,003 - INFO - Epoch 1/5, Batch 83/144 Batch Loss: 0.0906
2024-06-22 18:27:08,525 - INFO - Epoch 1/5, Batch 84/144 Batch Loss: 0.0937
2024-06-22 18:27:10,044 - INFO - Epoch 1/5, Batch 85/144 Batch Loss: 0.0835
2024-06-22 18:27:11,565 - INFO - Epoch 1/5, Batch 86/144 Batch Loss: 0.0895
2024-06-22 18:27:13,088 - INFO - Epoch 1/5, Batch 87/144 Batch Loss: 0.0962
2024-06-22 18:27:14,609 - INFO - Epoch 1/5, Batch 88/144 Batch Loss: 0.0848
2024-06-22 18:27:16,132 - INFO - Epoch 1/5, Batch 89/144 Batch Loss: 0.0879
2024-06-22 18:27:17,680 - INFO - Epoch 1/5, Batch 90/144 Batch Loss: 0.0924
2024-06-22 18:27:19,203 - INFO - Epoch 1/5, Batch 91/144 Batch Loss: 0.0855
2024-06-22 18:27:20,734 - INFO - Epoch 1/5, Batch 92/144 Batch Loss: 0.0795
2024-06-22 18:27:22,268 - INFO - Epoch 1/5, Batch 93/144 Batch Loss: 0.0820
2024-06-22 18:27:23,808 - INFO - Epoch 1/5, Batch 94/144 Batch Loss: 0.0838
2024-06-22 18:27:25,330 - INFO - Epoch 1/5, Batch 95/144 Batch Loss: 0.0873
2024-06-22 18:27:26,851 - INFO - Epoch 1/5, Batch 96/144 Batch Loss: 0.0953
2024-06-22 18:27:28,383 - INFO - Epoch 1/5, Batch 97/144 Batch Loss: 0.0901
2024-06-22 18:27:29,906 - INFO - Epoch 1/5, Batch 98/144 Batch Loss: 0.0856
2024-06-22 18:27:31,426 - INFO - Epoch 1/5, Batch 99/144 Batch Loss: 0.0946
2024-06-22 18:27:32,950 - INFO - Epoch 1/5, Batch 100/144 Batch Loss: 0.0838
2024-06-22 18:27:34,473 - INFO - Epoch 1/5, Batch 101/144 Batch Loss: 0.0767
2024-06-22 18:27:35,991 - INFO - Epoch 1/5, Batch 102/144 Batch Loss: 0.0982
2024-06-22 18:27:37,539 - INFO - Epoch 1/5, Batch 103/144 Batch Loss: 0.0899
2024-06-22 18:27:39,064 - INFO - Epoch 1/5, Batch 104/144 Batch Loss: 0.0869
2024-06-22 18:27:40,582 - INFO - Epoch 1/5, Batch 105/144 Batch Loss: 0.0900
2024-06-22 18:27:42,108 - INFO - Epoch 1/5, Batch 106/144 Batch Loss: 0.0846
2024-06-22 18:27:43,632 - INFO - Epoch 1/5, Batch 107/144 Batch Loss: 0.0913
2024-06-22 18:27:45,155 - INFO - Epoch 1/5, Batch 108/144 Batch Loss: 0.0842
2024-06-22 18:27:46,679 - INFO - Epoch 1/5, Batch 109/144 Batch Loss: 0.0845
2024-06-22 18:27:48,231 - INFO - Epoch 1/5, Batch 110/144 Batch Loss: 0.0764
2024-06-22 18:27:49,751 - INFO - Epoch 1/5, Batch 111/144 Batch Loss: 0.0855
2024-06-22 18:27:51,276 - INFO - Epoch 1/5, Batch 112/144 Batch Loss: 0.0922
2024-06-22 18:27:52,802 - INFO - Epoch 1/5, Batch 113/144 Batch Loss: 0.0791
2024-06-22 18:27:54,324 - INFO - Epoch 1/5, Batch 114/144 Batch Loss: 0.0787
2024-06-22 18:27:55,857 - INFO - Epoch 1/5, Batch 115/144 Batch Loss: 0.0878
2024-06-22 18:27:57,433 - INFO - Epoch 1/5, Batch 116/144 Batch Loss: 0.0836
2024-06-22 18:27:58,972 - INFO - Epoch 1/5, Batch 117/144 Batch Loss: 0.0846
2024-06-22 18:28:00,512 - INFO - Epoch 1/5, Batch 118/144 Batch Loss: 0.0811
2024-06-22 18:28:02,056 - INFO - Epoch 1/5, Batch 119/144 Batch Loss: 0.0795
2024-06-22 18:28:03,600 - INFO - Epoch 1/5, Batch 120/144 Batch Loss: 0.0844
2024-06-22 18:28:05,148 - INFO - Epoch 1/5, Batch 121/144 Batch Loss: 0.0797
2024-06-22 18:28:06,726 - INFO - Epoch 1/5, Batch 122/144 Batch Loss: 0.0836
2024-06-22 18:28:08,288 - INFO - Epoch 1/5, Batch 123/144 Batch Loss: 0.0810
2024-06-22 18:28:09,844 - INFO - Epoch 1/5, Batch 124/144 Batch Loss: 0.0876
2024-06-22 18:28:11,405 - INFO - Epoch 1/5, Batch 125/144 Batch Loss: 0.0780
2024-06-22 18:28:12,964 - INFO - Epoch 1/5, Batch 126/144 Batch Loss: 0.0846
2024-06-22 18:28:14,530 - INFO - Epoch 1/5, Batch 127/144 Batch Loss: 0.0800
2024-06-22 18:28:16,111 - INFO - Epoch 1/5, Batch 128/144 Batch Loss: 0.0889
2024-06-22 18:28:17,685 - INFO - Epoch 1/5, Batch 129/144 Batch Loss: 0.0853
2024-06-22 18:28:19,290 - INFO - Epoch 1/5, Batch 130/144 Batch Loss: 0.0727
2024-06-22 18:28:20,871 - INFO - Epoch 1/5, Batch 131/144 Batch Loss: 0.0769
2024-06-22 18:28:22,449 - INFO - Epoch 1/5, Batch 132/144 Batch Loss: 0.0708
2024-06-22 18:28:24,034 - INFO - Epoch 1/5, Batch 133/144 Batch Loss: 0.0643
2024-06-22 18:28:25,618 - INFO - Epoch 1/5, Batch 134/144 Batch Loss: 0.0646
2024-06-22 18:28:27,211 - INFO - Epoch 1/5, Batch 135/144 Batch Loss: 0.0576
2024-06-22 18:28:28,805 - INFO - Epoch 1/5, Batch 136/144 Batch Loss: 0.0522
2024-06-22 18:28:30,407 - INFO - Epoch 1/5, Batch 137/144 Batch Loss: 0.0458
2024-06-22 18:28:32,005 - INFO - Epoch 1/5, Batch 138/144 Batch Loss: 0.0410
2024-06-22 18:28:33,650 - INFO - Epoch 1/5, Batch 139/144 Batch Loss: 0.0415
2024-06-22 18:28:35,259 - INFO - Epoch 1/5, Batch 140/144 Batch Loss: 0.0425
2024-06-22 18:28:36,889 - INFO - Epoch 1/5, Batch 141/144 Batch Loss: 0.0422
2024-06-22 18:28:38,508 - INFO - Epoch 1/5, Batch 142/144 Batch Loss: 0.0391
2024-06-22 18:28:40,132 - INFO - Epoch 1/5, Batch 143/144 Batch Loss: 0.0434
2024-06-22 18:28:41,758 - INFO - Epoch 1/5, Batch 144/144 Batch Loss: 0.0372
2024-06-22 18:29:17,845 - INFO - Epoch 1/5 - Train Loss: 0.0361, Valid Loss: 0.0412, 
2024-06-22 18:29:19,554 - INFO - Epoch 2/5, Batch 1/144 Batch Loss: 0.0363
2024-06-22 18:29:21,205 - INFO - Epoch 2/5, Batch 2/144 Batch Loss: 0.0359
2024-06-22 18:29:22,882 - INFO - Epoch 2/5, Batch 3/144 Batch Loss: 0.0348
2024-06-22 18:29:24,527 - INFO - Epoch 2/5, Batch 4/144 Batch Loss: 0.0329
2024-06-22 18:29:26,207 - INFO - Epoch 2/5, Batch 5/144 Batch Loss: 0.0314
2024-06-22 18:29:27,850 - INFO - Epoch 2/5, Batch 6/144 Batch Loss: 0.0309
2024-06-22 18:29:29,424 - INFO - Epoch 2/5, Batch 7/144 Batch Loss: 0.0312
2024-06-22 18:29:30,960 - INFO - Epoch 2/5, Batch 8/144 Batch Loss: 0.0304
2024-06-22 18:29:32,490 - INFO - Epoch 2/5, Batch 9/144 Batch Loss: 0.0293
2024-06-22 18:29:34,107 - INFO - Epoch 2/5, Batch 10/144 Batch Loss: 0.0282
2024-06-22 18:29:35,657 - INFO - Epoch 2/5, Batch 11/144 Batch Loss: 0.0273
2024-06-22 18:29:37,243 - INFO - Epoch 2/5, Batch 12/144 Batch Loss: 0.0267
2024-06-22 18:29:38,793 - INFO - Epoch 2/5, Batch 13/144 Batch Loss: 0.0276
2024-06-22 18:29:40,382 - INFO - Epoch 2/5, Batch 14/144 Batch Loss: 0.0260
2024-06-22 18:29:41,932 - INFO - Epoch 2/5, Batch 15/144 Batch Loss: 0.0254
2024-06-22 18:29:43,555 - INFO - Epoch 2/5, Batch 16/144 Batch Loss: 0.0249
2024-06-22 18:29:45,105 - INFO - Epoch 2/5, Batch 17/144 Batch Loss: 0.0246
2024-06-22 18:29:46,703 - INFO - Epoch 2/5, Batch 18/144 Batch Loss: 0.0244
2024-06-22 18:29:48,251 - INFO - Epoch 2/5, Batch 19/144 Batch Loss: 0.0228
2024-06-22 18:29:49,852 - INFO - Epoch 2/5, Batch 20/144 Batch Loss: 0.0234
2024-06-22 18:29:51,400 - INFO - Epoch 2/5, Batch 21/144 Batch Loss: 0.0214
2024-06-22 18:29:53,004 - INFO - Epoch 2/5, Batch 22/144 Batch Loss: 0.0222
2024-06-22 18:29:54,571 - INFO - Epoch 2/5, Batch 23/144 Batch Loss: 0.0219
2024-06-22 18:29:56,194 - INFO - Epoch 2/5, Batch 24/144 Batch Loss: 0.0219
2024-06-22 18:29:57,747 - INFO - Epoch 2/5, Batch 25/144 Batch Loss: 0.0211
2024-06-22 18:29:59,353 - INFO - Epoch 2/5, Batch 26/144 Batch Loss: 0.0214
2024-06-22 18:30:00,904 - INFO - Epoch 2/5, Batch 27/144 Batch Loss: 0.0203
2024-06-22 18:30:02,502 - INFO - Epoch 2/5, Batch 28/144 Batch Loss: 0.0202
2024-06-22 18:30:04,069 - INFO - Epoch 2/5, Batch 29/144 Batch Loss: 0.0206
2024-06-22 18:30:05,662 - INFO - Epoch 2/5, Batch 30/144 Batch Loss: 0.0195
2024-06-22 18:30:07,221 - INFO - Epoch 2/5, Batch 31/144 Batch Loss: 0.0198
2024-06-22 18:30:08,837 - INFO - Epoch 2/5, Batch 32/144 Batch Loss: 0.0204
2024-06-22 18:30:10,418 - INFO - Epoch 2/5, Batch 33/144 Batch Loss: 0.0182
2024-06-22 18:30:12,020 - INFO - Epoch 2/5, Batch 34/144 Batch Loss: 0.0194
2024-06-22 18:30:13,579 - INFO - Epoch 2/5, Batch 35/144 Batch Loss: 0.0196
2024-06-22 18:30:15,186 - INFO - Epoch 2/5, Batch 36/144 Batch Loss: 0.0195
2024-06-22 18:30:16,756 - INFO - Epoch 2/5, Batch 37/144 Batch Loss: 0.0182
2024-06-22 18:30:18,364 - INFO - Epoch 2/5, Batch 38/144 Batch Loss: 0.0183
2024-06-22 18:30:19,925 - INFO - Epoch 2/5, Batch 39/144 Batch Loss: 0.0186
2024-06-22 18:30:21,539 - INFO - Epoch 2/5, Batch 40/144 Batch Loss: 0.0180
2024-06-22 18:30:23,103 - INFO - Epoch 2/5, Batch 41/144 Batch Loss: 0.0187
2024-06-22 18:30:24,719 - INFO - Epoch 2/5, Batch 42/144 Batch Loss: 0.0184
2024-06-22 18:30:26,314 - INFO - Epoch 2/5, Batch 43/144 Batch Loss: 0.0175
2024-06-22 18:30:27,933 - INFO - Epoch 2/5, Batch 44/144 Batch Loss: 0.0172
2024-06-22 18:30:29,495 - INFO - Epoch 2/5, Batch 45/144 Batch Loss: 0.0171
2024-06-22 18:30:31,122 - INFO - Epoch 2/5, Batch 46/144 Batch Loss: 0.0176
2024-06-22 18:30:32,692 - INFO - Epoch 2/5, Batch 47/144 Batch Loss: 0.0171
2024-06-22 18:30:34,315 - INFO - Epoch 2/5, Batch 48/144 Batch Loss: 0.0176
2024-06-22 18:30:35,885 - INFO - Epoch 2/5, Batch 49/144 Batch Loss: 0.0172
2024-06-22 18:30:37,510 - INFO - Epoch 2/5, Batch 50/144 Batch Loss: 0.0179
2024-06-22 18:30:39,083 - INFO - Epoch 2/5, Batch 51/144 Batch Loss: 0.0176
2024-06-22 18:30:40,699 - INFO - Epoch 2/5, Batch 52/144 Batch Loss: 0.0162
2024-06-22 18:30:42,279 - INFO - Epoch 2/5, Batch 53/144 Batch Loss: 0.0179
2024-06-22 18:30:43,899 - INFO - Epoch 2/5, Batch 54/144 Batch Loss: 0.0165
2024-06-22 18:30:45,473 - INFO - Epoch 2/5, Batch 55/144 Batch Loss: 0.0162
2024-06-22 18:30:47,130 - INFO - Epoch 2/5, Batch 56/144 Batch Loss: 0.0171
2024-06-22 18:30:48,715 - INFO - Epoch 2/5, Batch 57/144 Batch Loss: 0.0164
2024-06-22 18:30:50,336 - INFO - Epoch 2/5, Batch 58/144 Batch Loss: 0.0159
2024-06-22 18:30:51,917 - INFO - Epoch 2/5, Batch 59/144 Batch Loss: 0.0169
2024-06-22 18:30:53,541 - INFO - Epoch 2/5, Batch 60/144 Batch Loss: 0.0159
2024-06-22 18:30:55,130 - INFO - Epoch 2/5, Batch 61/144 Batch Loss: 0.0173
2024-06-22 18:30:56,756 - INFO - Epoch 2/5, Batch 62/144 Batch Loss: 0.0166
2024-06-22 18:30:58,345 - INFO - Epoch 2/5, Batch 63/144 Batch Loss: 0.0157
2024-06-22 18:30:59,972 - INFO - Epoch 2/5, Batch 64/144 Batch Loss: 0.0157
2024-06-22 18:31:01,559 - INFO - Epoch 2/5, Batch 65/144 Batch Loss: 0.0165
2024-06-22 18:31:03,189 - INFO - Epoch 2/5, Batch 66/144 Batch Loss: 0.0159
2024-06-22 18:31:04,779 - INFO - Epoch 2/5, Batch 67/144 Batch Loss: 0.0158
2024-06-22 18:31:06,412 - INFO - Epoch 2/5, Batch 68/144 Batch Loss: 0.0152
2024-06-22 18:31:07,996 - INFO - Epoch 2/5, Batch 69/144 Batch Loss: 0.0160
2024-06-22 18:31:09,631 - INFO - Epoch 2/5, Batch 70/144 Batch Loss: 0.0160
2024-06-22 18:31:11,216 - INFO - Epoch 2/5, Batch 71/144 Batch Loss: 0.0157
2024-06-22 18:31:12,891 - INFO - Epoch 2/5, Batch 72/144 Batch Loss: 0.0152
2024-06-22 18:31:14,480 - INFO - Epoch 2/5, Batch 73/144 Batch Loss: 0.0167
2024-06-22 18:31:16,030 - INFO - Epoch 2/5, Batch 74/144 Batch Loss: 0.0161
2024-06-22 18:31:17,586 - INFO - Epoch 2/5, Batch 75/144 Batch Loss: 0.0153
2024-06-22 18:31:19,126 - INFO - Epoch 2/5, Batch 76/144 Batch Loss: 0.0160
2024-06-22 18:31:20,670 - INFO - Epoch 2/5, Batch 77/144 Batch Loss: 0.0149
2024-06-22 18:31:22,202 - INFO - Epoch 2/5, Batch 78/144 Batch Loss: 0.0161
2024-06-22 18:31:23,783 - INFO - Epoch 2/5, Batch 79/144 Batch Loss: 0.0155
2024-06-22 18:31:25,306 - INFO - Epoch 2/5, Batch 80/144 Batch Loss: 0.0149
2024-06-22 18:31:26,834 - INFO - Epoch 2/5, Batch 81/144 Batch Loss: 0.0151
2024-06-22 18:31:28,362 - INFO - Epoch 2/5, Batch 82/144 Batch Loss: 0.0149
2024-06-22 18:31:29,886 - INFO - Epoch 2/5, Batch 83/144 Batch Loss: 0.0154
2024-06-22 18:31:31,420 - INFO - Epoch 2/5, Batch 84/144 Batch Loss: 0.0158
2024-06-22 18:31:32,977 - INFO - Epoch 2/5, Batch 85/144 Batch Loss: 0.0158
2024-06-22 18:31:34,505 - INFO - Epoch 2/5, Batch 86/144 Batch Loss: 0.0149
2024-06-22 18:31:36,027 - INFO - Epoch 2/5, Batch 87/144 Batch Loss: 0.0155
2024-06-22 18:31:37,553 - INFO - Epoch 2/5, Batch 88/144 Batch Loss: 0.0149
2024-06-22 18:31:39,073 - INFO - Epoch 2/5, Batch 89/144 Batch Loss: 0.0156
2024-06-22 18:31:40,602 - INFO - Epoch 2/5, Batch 90/144 Batch Loss: 0.0153
2024-06-22 18:31:42,126 - INFO - Epoch 2/5, Batch 91/144 Batch Loss: 0.0154
2024-06-22 18:31:43,644 - INFO - Epoch 2/5, Batch 92/144 Batch Loss: 0.0149
2024-06-22 18:31:45,221 - INFO - Epoch 2/5, Batch 93/144 Batch Loss: 0.0143
2024-06-22 18:31:46,752 - INFO - Epoch 2/5, Batch 94/144 Batch Loss: 0.0143
2024-06-22 18:31:48,285 - INFO - Epoch 2/5, Batch 95/144 Batch Loss: 0.0137
2024-06-22 18:31:49,814 - INFO - Epoch 2/5, Batch 96/144 Batch Loss: 0.0139
2024-06-22 18:31:51,349 - INFO - Epoch 2/5, Batch 97/144 Batch Loss: 0.0145
2024-06-22 18:31:52,873 - INFO - Epoch 2/5, Batch 98/144 Batch Loss: 0.0142
2024-06-22 18:31:54,405 - INFO - Epoch 2/5, Batch 99/144 Batch Loss: 0.0145
2024-06-22 18:31:55,932 - INFO - Epoch 2/5, Batch 100/144 Batch Loss: 0.0149
2024-06-22 18:31:57,471 - INFO - Epoch 2/5, Batch 101/144 Batch Loss: 0.0146
2024-06-22 18:31:59,025 - INFO - Epoch 2/5, Batch 102/144 Batch Loss: 0.0150
2024-06-22 18:32:00,565 - INFO - Epoch 2/5, Batch 103/144 Batch Loss: 0.0147
2024-06-22 18:32:02,095 - INFO - Epoch 2/5, Batch 104/144 Batch Loss: 0.0142
2024-06-22 18:32:03,635 - INFO - Epoch 2/5, Batch 105/144 Batch Loss: 0.0143
2024-06-22 18:32:05,178 - INFO - Epoch 2/5, Batch 106/144 Batch Loss: 0.0158
2024-06-22 18:32:06,705 - INFO - Epoch 2/5, Batch 107/144 Batch Loss: 0.0146
2024-06-22 18:32:08,244 - INFO - Epoch 2/5, Batch 108/144 Batch Loss: 0.0145
2024-06-22 18:32:09,808 - INFO - Epoch 2/5, Batch 109/144 Batch Loss: 0.0135
2024-06-22 18:32:11,343 - INFO - Epoch 2/5, Batch 110/144 Batch Loss: 0.0142
2024-06-22 18:32:12,872 - INFO - Epoch 2/5, Batch 111/144 Batch Loss: 0.0149
2024-06-22 18:32:14,398 - INFO - Epoch 2/5, Batch 112/144 Batch Loss: 0.0146
2024-06-22 18:32:15,921 - INFO - Epoch 2/5, Batch 113/144 Batch Loss: 0.0147
2024-06-22 18:32:17,462 - INFO - Epoch 2/5, Batch 114/144 Batch Loss: 0.0134
2024-06-22 18:32:19,005 - INFO - Epoch 2/5, Batch 115/144 Batch Loss: 0.0147
2024-06-22 18:32:20,554 - INFO - Epoch 2/5, Batch 116/144 Batch Loss: 0.0137
2024-06-22 18:32:22,082 - INFO - Epoch 2/5, Batch 117/144 Batch Loss: 0.0143
2024-06-22 18:32:23,607 - INFO - Epoch 2/5, Batch 118/144 Batch Loss: 0.0138
2024-06-22 18:32:25,141 - INFO - Epoch 2/5, Batch 119/144 Batch Loss: 0.0132
2024-06-22 18:32:26,665 - INFO - Epoch 2/5, Batch 120/144 Batch Loss: 0.0139
2024-06-22 18:32:28,188 - INFO - Epoch 2/5, Batch 121/144 Batch Loss: 0.0152
2024-06-22 18:32:29,711 - INFO - Epoch 2/5, Batch 122/144 Batch Loss: 0.0138
2024-06-22 18:32:31,270 - INFO - Epoch 2/5, Batch 123/144 Batch Loss: 0.0141
2024-06-22 18:32:32,802 - INFO - Epoch 2/5, Batch 124/144 Batch Loss: 0.0137
2024-06-22 18:32:34,326 - INFO - Epoch 2/5, Batch 125/144 Batch Loss: 0.0133
2024-06-22 18:32:35,857 - INFO - Epoch 2/5, Batch 126/144 Batch Loss: 0.0140
2024-06-22 18:32:37,389 - INFO - Epoch 2/5, Batch 127/144 Batch Loss: 0.0139
2024-06-22 18:32:38,913 - INFO - Epoch 2/5, Batch 128/144 Batch Loss: 0.0148
2024-06-22 18:32:40,439 - INFO - Epoch 2/5, Batch 129/144 Batch Loss: 0.0131
2024-06-22 18:32:41,975 - INFO - Epoch 2/5, Batch 130/144 Batch Loss: 0.0142
2024-06-22 18:32:43,498 - INFO - Epoch 2/5, Batch 131/144 Batch Loss: 0.0130
2024-06-22 18:32:45,050 - INFO - Epoch 2/5, Batch 132/144 Batch Loss: 0.0136
2024-06-22 18:32:46,593 - INFO - Epoch 2/5, Batch 133/144 Batch Loss: 0.0133
2024-06-22 18:32:48,124 - INFO - Epoch 2/5, Batch 134/144 Batch Loss: 0.0130
2024-06-22 18:32:49,654 - INFO - Epoch 2/5, Batch 135/144 Batch Loss: 0.0142
2024-06-22 18:32:51,192 - INFO - Epoch 2/5, Batch 136/144 Batch Loss: 0.0133
2024-06-22 18:32:52,716 - INFO - Epoch 2/5, Batch 137/144 Batch Loss: 0.0145
2024-06-22 18:32:54,250 - INFO - Epoch 2/5, Batch 138/144 Batch Loss: 0.0139
2024-06-22 18:32:55,777 - INFO - Epoch 2/5, Batch 139/144 Batch Loss: 0.0146
2024-06-22 18:32:57,303 - INFO - Epoch 2/5, Batch 140/144 Batch Loss: 0.0137
2024-06-22 18:32:58,841 - INFO - Epoch 2/5, Batch 141/144 Batch Loss: 0.0144
2024-06-22 18:33:00,366 - INFO - Epoch 2/5, Batch 142/144 Batch Loss: 0.0143
2024-06-22 18:33:01,903 - INFO - Epoch 2/5, Batch 143/144 Batch Loss: 0.0128
2024-06-22 18:33:03,459 - INFO - Epoch 2/5, Batch 144/144 Batch Loss: 0.0128
2024-06-22 18:33:35,130 - INFO - Epoch 2/5 - Train Loss: 0.0136, Valid Loss: 0.0192, 
2024-06-22 18:33:36,675 - INFO - Epoch 3/5, Batch 1/144 Batch Loss: 0.0132
2024-06-22 18:33:38,200 - INFO - Epoch 3/5, Batch 2/144 Batch Loss: 0.0143
2024-06-22 18:33:39,738 - INFO - Epoch 3/5, Batch 3/144 Batch Loss: 0.0144
2024-06-22 18:33:41,270 - INFO - Epoch 3/5, Batch 4/144 Batch Loss: 0.0134
2024-06-22 18:33:42,798 - INFO - Epoch 3/5, Batch 5/144 Batch Loss: 0.0138
2024-06-22 18:33:44,353 - INFO - Epoch 3/5, Batch 6/144 Batch Loss: 0.0153
2024-06-22 18:33:45,879 - INFO - Epoch 3/5, Batch 7/144 Batch Loss: 0.0138
2024-06-22 18:33:47,408 - INFO - Epoch 3/5, Batch 8/144 Batch Loss: 0.0135
2024-06-22 18:33:48,963 - INFO - Epoch 3/5, Batch 9/144 Batch Loss: 0.0130
2024-06-22 18:33:50,500 - INFO - Epoch 3/5, Batch 10/144 Batch Loss: 0.0142
2024-06-22 18:33:52,028 - INFO - Epoch 3/5, Batch 11/144 Batch Loss: 0.0131
2024-06-22 18:33:53,562 - INFO - Epoch 3/5, Batch 12/144 Batch Loss: 0.0135
2024-06-22 18:33:55,094 - INFO - Epoch 3/5, Batch 13/144 Batch Loss: 0.0133
2024-06-22 18:33:56,630 - INFO - Epoch 3/5, Batch 14/144 Batch Loss: 0.0129
2024-06-22 18:33:58,170 - INFO - Epoch 3/5, Batch 15/144 Batch Loss: 0.0135
2024-06-22 18:33:59,716 - INFO - Epoch 3/5, Batch 16/144 Batch Loss: 0.0132
2024-06-22 18:34:01,276 - INFO - Epoch 3/5, Batch 17/144 Batch Loss: 0.0129
2024-06-22 18:34:02,820 - INFO - Epoch 3/5, Batch 18/144 Batch Loss: 0.0133
2024-06-22 18:34:04,373 - INFO - Epoch 3/5, Batch 19/144 Batch Loss: 0.0126
2024-06-22 18:34:05,926 - INFO - Epoch 3/5, Batch 20/144 Batch Loss: 0.0133
2024-06-22 18:34:07,486 - INFO - Epoch 3/5, Batch 21/144 Batch Loss: 0.0132
2024-06-22 18:34:09,066 - INFO - Epoch 3/5, Batch 22/144 Batch Loss: 0.0141
2024-06-22 18:34:10,630 - INFO - Epoch 3/5, Batch 23/144 Batch Loss: 0.0135
2024-06-22 18:34:12,204 - INFO - Epoch 3/5, Batch 24/144 Batch Loss: 0.0143
2024-06-22 18:34:13,768 - INFO - Epoch 3/5, Batch 25/144 Batch Loss: 0.0138
2024-06-22 18:34:15,338 - INFO - Epoch 3/5, Batch 26/144 Batch Loss: 0.0133
2024-06-22 18:34:16,916 - INFO - Epoch 3/5, Batch 27/144 Batch Loss: 0.0135
2024-06-22 18:34:18,487 - INFO - Epoch 3/5, Batch 28/144 Batch Loss: 0.0141
2024-06-22 18:34:20,075 - INFO - Epoch 3/5, Batch 29/144 Batch Loss: 0.0136
2024-06-22 18:34:21,667 - INFO - Epoch 3/5, Batch 30/144 Batch Loss: 0.0128
2024-06-22 18:34:23,278 - INFO - Epoch 3/5, Batch 31/144 Batch Loss: 0.0127
2024-06-22 18:34:24,860 - INFO - Epoch 3/5, Batch 32/144 Batch Loss: 0.0136
2024-06-22 18:34:26,458 - INFO - Epoch 3/5, Batch 33/144 Batch Loss: 0.0137
2024-06-22 18:34:28,046 - INFO - Epoch 3/5, Batch 34/144 Batch Loss: 0.0143
2024-06-22 18:34:29,635 - INFO - Epoch 3/5, Batch 35/144 Batch Loss: 0.0139
2024-06-22 18:34:31,226 - INFO - Epoch 3/5, Batch 36/144 Batch Loss: 0.0135
2024-06-22 18:34:32,825 - INFO - Epoch 3/5, Batch 37/144 Batch Loss: 0.0129
2024-06-22 18:34:34,422 - INFO - Epoch 3/5, Batch 38/144 Batch Loss: 0.0131
2024-06-22 18:34:36,020 - INFO - Epoch 3/5, Batch 39/144 Batch Loss: 0.0133
2024-06-22 18:34:37,613 - INFO - Epoch 3/5, Batch 40/144 Batch Loss: 0.0130
2024-06-22 18:34:39,218 - INFO - Epoch 3/5, Batch 41/144 Batch Loss: 0.0144
2024-06-22 18:34:40,839 - INFO - Epoch 3/5, Batch 42/144 Batch Loss: 0.0119
2024-06-22 18:34:42,449 - INFO - Epoch 3/5, Batch 43/144 Batch Loss: 0.0123
2024-06-22 18:34:44,056 - INFO - Epoch 3/5, Batch 44/144 Batch Loss: 0.0142
2024-06-22 18:34:45,665 - INFO - Epoch 3/5, Batch 45/144 Batch Loss: 0.0121
2024-06-22 18:34:47,276 - INFO - Epoch 3/5, Batch 46/144 Batch Loss: 0.0129
2024-06-22 18:34:48,894 - INFO - Epoch 3/5, Batch 47/144 Batch Loss: 0.0125
2024-06-22 18:34:50,508 - INFO - Epoch 3/5, Batch 48/144 Batch Loss: 0.0128
2024-06-22 18:34:52,131 - INFO - Epoch 3/5, Batch 49/144 Batch Loss: 0.0115
2024-06-22 18:34:53,751 - INFO - Epoch 3/5, Batch 50/144 Batch Loss: 0.0136
2024-06-22 18:34:55,396 - INFO - Epoch 3/5, Batch 51/144 Batch Loss: 0.0145
2024-06-22 18:34:57,039 - INFO - Epoch 3/5, Batch 52/144 Batch Loss: 0.0139
2024-06-22 18:34:58,727 - INFO - Epoch 3/5, Batch 53/144 Batch Loss: 0.0134
2024-06-22 18:35:00,351 - INFO - Epoch 3/5, Batch 54/144 Batch Loss: 0.0131
2024-06-22 18:35:01,982 - INFO - Epoch 3/5, Batch 55/144 Batch Loss: 0.0128
2024-06-22 18:35:03,614 - INFO - Epoch 3/5, Batch 56/144 Batch Loss: 0.0130
2024-06-22 18:35:05,269 - INFO - Epoch 3/5, Batch 57/144 Batch Loss: 0.0131
2024-06-22 18:35:06,899 - INFO - Epoch 3/5, Batch 58/144 Batch Loss: 0.0128
2024-06-22 18:35:08,529 - INFO - Epoch 3/5, Batch 59/144 Batch Loss: 0.0127
2024-06-22 18:35:10,169 - INFO - Epoch 3/5, Batch 60/144 Batch Loss: 0.0131
2024-06-22 18:35:11,809 - INFO - Epoch 3/5, Batch 61/144 Batch Loss: 0.0126
2024-06-22 18:35:13,459 - INFO - Epoch 3/5, Batch 62/144 Batch Loss: 0.0139
2024-06-22 18:35:15,101 - INFO - Epoch 3/5, Batch 63/144 Batch Loss: 0.0131
2024-06-22 18:35:16,739 - INFO - Epoch 3/5, Batch 64/144 Batch Loss: 0.0125
2024-06-22 18:35:18,415 - INFO - Epoch 3/5, Batch 65/144 Batch Loss: 0.0128
2024-06-22 18:35:20,065 - INFO - Epoch 3/5, Batch 66/144 Batch Loss: 0.0139
2024-06-22 18:35:21,715 - INFO - Epoch 3/5, Batch 67/144 Batch Loss: 0.0133
2024-06-22 18:35:23,362 - INFO - Epoch 3/5, Batch 68/144 Batch Loss: 0.0125
2024-06-22 18:35:25,012 - INFO - Epoch 3/5, Batch 69/144 Batch Loss: 0.0135
2024-06-22 18:35:26,667 - INFO - Epoch 3/5, Batch 70/144 Batch Loss: 0.0125
2024-06-22 18:35:28,358 - INFO - Epoch 3/5, Batch 71/144 Batch Loss: 0.0127
2024-06-22 18:35:30,009 - INFO - Epoch 3/5, Batch 72/144 Batch Loss: 0.0121
2024-06-22 18:35:31,669 - INFO - Epoch 3/5, Batch 73/144 Batch Loss: 0.0121
2024-06-22 18:35:33,327 - INFO - Epoch 3/5, Batch 74/144 Batch Loss: 0.0128
2024-06-22 18:35:34,985 - INFO - Epoch 3/5, Batch 75/144 Batch Loss: 0.0138
2024-06-22 18:35:36,644 - INFO - Epoch 3/5, Batch 76/144 Batch Loss: 0.0139
2024-06-22 18:35:38,345 - INFO - Epoch 3/5, Batch 77/144 Batch Loss: 0.0133
2024-06-22 18:35:40,003 - INFO - Epoch 3/5, Batch 78/144 Batch Loss: 0.0133
2024-06-22 18:35:41,670 - INFO - Epoch 3/5, Batch 79/144 Batch Loss: 0.0143
2024-06-22 18:35:43,331 - INFO - Epoch 3/5, Batch 80/144 Batch Loss: 0.0132
2024-06-22 18:35:45,000 - INFO - Epoch 3/5, Batch 81/144 Batch Loss: 0.0129
2024-06-22 18:35:46,663 - INFO - Epoch 3/5, Batch 82/144 Batch Loss: 0.0124
2024-06-22 18:35:48,330 - INFO - Epoch 3/5, Batch 83/144 Batch Loss: 0.0122
2024-06-22 18:35:50,037 - INFO - Epoch 3/5, Batch 84/144 Batch Loss: 0.0124
2024-06-22 18:35:51,711 - INFO - Epoch 3/5, Batch 85/144 Batch Loss: 0.0124
2024-06-22 18:35:53,383 - INFO - Epoch 3/5, Batch 86/144 Batch Loss: 0.0135
2024-06-22 18:35:55,055 - INFO - Epoch 3/5, Batch 87/144 Batch Loss: 0.0140
2024-06-22 18:35:56,763 - INFO - Epoch 3/5, Batch 88/144 Batch Loss: 0.0127
2024-06-22 18:35:58,440 - INFO - Epoch 3/5, Batch 89/144 Batch Loss: 0.0124
2024-06-22 18:36:00,120 - INFO - Epoch 3/5, Batch 90/144 Batch Loss: 0.0125
2024-06-22 18:36:01,819 - INFO - Epoch 3/5, Batch 91/144 Batch Loss: 0.0135
2024-06-22 18:36:03,526 - INFO - Epoch 3/5, Batch 92/144 Batch Loss: 0.0133
2024-06-22 18:36:05,206 - INFO - Epoch 3/5, Batch 93/144 Batch Loss: 0.0127
2024-06-22 18:36:06,890 - INFO - Epoch 3/5, Batch 94/144 Batch Loss: 0.0131
2024-06-22 18:36:08,576 - INFO - Epoch 3/5, Batch 95/144 Batch Loss: 0.0137
2024-06-22 18:36:10,261 - INFO - Epoch 3/5, Batch 96/144 Batch Loss: 0.0132
2024-06-22 18:36:11,952 - INFO - Epoch 3/5, Batch 97/144 Batch Loss: 0.0148
2024-06-22 18:36:13,669 - INFO - Epoch 3/5, Batch 98/144 Batch Loss: 0.0136
2024-06-22 18:36:15,360 - INFO - Epoch 3/5, Batch 99/144 Batch Loss: 0.0126
2024-06-22 18:36:17,054 - INFO - Epoch 3/5, Batch 100/144 Batch Loss: 0.0130
2024-06-22 18:36:18,741 - INFO - Epoch 3/5, Batch 101/144 Batch Loss: 0.0127
2024-06-22 18:36:20,432 - INFO - Epoch 3/5, Batch 102/144 Batch Loss: 0.0140
2024-06-22 18:36:22,123 - INFO - Epoch 3/5, Batch 103/144 Batch Loss: 0.0128
2024-06-22 18:36:23,822 - INFO - Epoch 3/5, Batch 104/144 Batch Loss: 0.0121
2024-06-22 18:36:25,557 - INFO - Epoch 3/5, Batch 105/144 Batch Loss: 0.0127
2024-06-22 18:36:27,257 - INFO - Epoch 3/5, Batch 106/144 Batch Loss: 0.0130
2024-06-22 18:36:28,957 - INFO - Epoch 3/5, Batch 107/144 Batch Loss: 0.0132
2024-06-22 18:36:30,660 - INFO - Epoch 3/5, Batch 108/144 Batch Loss: 0.0127
2024-06-22 18:36:32,369 - INFO - Epoch 3/5, Batch 109/144 Batch Loss: 0.0133
2024-06-22 18:36:34,073 - INFO - Epoch 3/5, Batch 110/144 Batch Loss: 0.0127
2024-06-22 18:36:35,775 - INFO - Epoch 3/5, Batch 111/144 Batch Loss: 0.0134
2024-06-22 18:36:37,478 - INFO - Epoch 3/5, Batch 112/144 Batch Loss: 0.0127
2024-06-22 18:36:39,222 - INFO - Epoch 3/5, Batch 113/144 Batch Loss: 0.0129
2024-06-22 18:36:40,929 - INFO - Epoch 3/5, Batch 114/144 Batch Loss: 0.0132
2024-06-22 18:36:42,638 - INFO - Epoch 3/5, Batch 115/144 Batch Loss: 0.0129
2024-06-22 18:36:44,346 - INFO - Epoch 3/5, Batch 116/144 Batch Loss: 0.0122
2024-06-22 18:36:46,053 - INFO - Epoch 3/5, Batch 117/144 Batch Loss: 0.0134
2024-06-22 18:36:47,763 - INFO - Epoch 3/5, Batch 118/144 Batch Loss: 0.0127
2024-06-22 18:36:49,510 - INFO - Epoch 3/5, Batch 119/144 Batch Loss: 0.0128
2024-06-22 18:36:51,227 - INFO - Epoch 3/5, Batch 120/144 Batch Loss: 0.0122
2024-06-22 18:36:52,944 - INFO - Epoch 3/5, Batch 121/144 Batch Loss: 0.0126
2024-06-22 18:36:54,658 - INFO - Epoch 3/5, Batch 122/144 Batch Loss: 0.0133
2024-06-22 18:36:56,371 - INFO - Epoch 3/5, Batch 123/144 Batch Loss: 0.0131
2024-06-22 18:36:58,171 - INFO - Epoch 3/5, Batch 124/144 Batch Loss: 0.0143
2024-06-22 18:36:59,894 - INFO - Epoch 3/5, Batch 125/144 Batch Loss: 0.0133
2024-06-22 18:37:01,614 - INFO - Epoch 3/5, Batch 126/144 Batch Loss: 0.0129
2024-06-22 18:37:03,234 - INFO - Epoch 3/5, Batch 127/144 Batch Loss: 0.0128
2024-06-22 18:37:04,941 - INFO - Epoch 3/5, Batch 128/144 Batch Loss: 0.0142
2024-06-22 18:37:06,664 - INFO - Epoch 3/5, Batch 129/144 Batch Loss: 0.0132
2024-06-22 18:37:08,393 - INFO - Epoch 3/5, Batch 130/144 Batch Loss: 0.0129
2024-06-22 18:37:10,139 - INFO - Epoch 3/5, Batch 131/144 Batch Loss: 0.0131
2024-06-22 18:37:11,863 - INFO - Epoch 3/5, Batch 132/144 Batch Loss: 0.0123
2024-06-22 18:37:13,477 - INFO - Epoch 3/5, Batch 133/144 Batch Loss: 0.0128
2024-06-22 18:37:15,185 - INFO - Epoch 3/5, Batch 134/144 Batch Loss: 0.0116
2024-06-22 18:37:16,917 - INFO - Epoch 3/5, Batch 135/144 Batch Loss: 0.0124
2024-06-22 18:37:18,686 - INFO - Epoch 3/5, Batch 136/144 Batch Loss: 0.0137
2024-06-22 18:37:20,418 - INFO - Epoch 3/5, Batch 137/144 Batch Loss: 0.0123
2024-06-22 18:37:22,147 - INFO - Epoch 3/5, Batch 138/144 Batch Loss: 0.0134
2024-06-22 18:37:23,872 - INFO - Epoch 3/5, Batch 139/144 Batch Loss: 0.0127
2024-06-22 18:37:25,606 - INFO - Epoch 3/5, Batch 140/144 Batch Loss: 0.0129
2024-06-22 18:37:27,338 - INFO - Epoch 3/5, Batch 141/144 Batch Loss: 0.0128
2024-06-22 18:37:29,109 - INFO - Epoch 3/5, Batch 142/144 Batch Loss: 0.0121
2024-06-22 18:37:30,839 - INFO - Epoch 3/5, Batch 143/144 Batch Loss: 0.0129
2024-06-22 18:37:32,569 - INFO - Epoch 3/5, Batch 144/144 Batch Loss: 0.0136
2024-06-22 18:38:10,475 - INFO - Epoch 3/5 - Train Loss: 0.0129, Valid Loss: 0.0184, 
2024-06-22 18:38:12,215 - INFO - Epoch 4/5, Batch 1/144 Batch Loss: 0.0121
2024-06-22 18:38:13,982 - INFO - Epoch 4/5, Batch 2/144 Batch Loss: 0.0127
2024-06-22 18:38:15,732 - INFO - Epoch 4/5, Batch 3/144 Batch Loss: 0.0130
2024-06-22 18:38:17,461 - INFO - Epoch 4/5, Batch 4/144 Batch Loss: 0.0130
2024-06-22 18:38:19,203 - INFO - Epoch 4/5, Batch 5/144 Batch Loss: 0.0126
2024-06-22 18:38:20,937 - INFO - Epoch 4/5, Batch 6/144 Batch Loss: 0.0126
2024-06-22 18:38:22,683 - INFO - Epoch 4/5, Batch 7/144 Batch Loss: 0.0125
2024-06-22 18:38:24,416 - INFO - Epoch 4/5, Batch 8/144 Batch Loss: 0.0122
2024-06-22 18:38:26,146 - INFO - Epoch 4/5, Batch 9/144 Batch Loss: 0.0121
2024-06-22 18:38:27,878 - INFO - Epoch 4/5, Batch 10/144 Batch Loss: 0.0126
2024-06-22 18:38:29,657 - INFO - Epoch 4/5, Batch 11/144 Batch Loss: 0.0127
2024-06-22 18:38:31,386 - INFO - Epoch 4/5, Batch 12/144 Batch Loss: 0.0135
2024-06-22 18:38:33,131 - INFO - Epoch 4/5, Batch 13/144 Batch Loss: 0.0131
2024-06-22 18:38:34,859 - INFO - Epoch 4/5, Batch 14/144 Batch Loss: 0.0128
2024-06-22 18:38:36,605 - INFO - Epoch 4/5, Batch 15/144 Batch Loss: 0.0121
2024-06-22 18:38:38,336 - INFO - Epoch 4/5, Batch 16/144 Batch Loss: 0.0129
2024-06-22 18:38:40,081 - INFO - Epoch 4/5, Batch 17/144 Batch Loss: 0.0130
2024-06-22 18:38:41,813 - INFO - Epoch 4/5, Batch 18/144 Batch Loss: 0.0135
2024-06-22 18:38:43,557 - INFO - Epoch 4/5, Batch 19/144 Batch Loss: 0.0124
2024-06-22 18:38:45,283 - INFO - Epoch 4/5, Batch 20/144 Batch Loss: 0.0121
2024-06-22 18:38:47,061 - INFO - Epoch 4/5, Batch 21/144 Batch Loss: 0.0129
2024-06-22 18:38:48,789 - INFO - Epoch 4/5, Batch 22/144 Batch Loss: 0.0134
2024-06-22 18:38:50,519 - INFO - Epoch 4/5, Batch 23/144 Batch Loss: 0.0131
2024-06-22 18:38:52,263 - INFO - Epoch 4/5, Batch 24/144 Batch Loss: 0.0123
2024-06-22 18:38:53,995 - INFO - Epoch 4/5, Batch 25/144 Batch Loss: 0.0127
2024-06-22 18:38:55,741 - INFO - Epoch 4/5, Batch 26/144 Batch Loss: 0.0125
2024-06-22 18:38:57,473 - INFO - Epoch 4/5, Batch 27/144 Batch Loss: 0.0125
2024-06-22 18:38:59,220 - INFO - Epoch 4/5, Batch 28/144 Batch Loss: 0.0135
2024-06-22 18:39:00,950 - INFO - Epoch 4/5, Batch 29/144 Batch Loss: 0.0128
2024-06-22 18:39:02,685 - INFO - Epoch 4/5, Batch 30/144 Batch Loss: 0.0134
2024-06-22 18:39:04,415 - INFO - Epoch 4/5, Batch 31/144 Batch Loss: 0.0124
2024-06-22 18:39:06,149 - INFO - Epoch 4/5, Batch 32/144 Batch Loss: 0.0126
2024-06-22 18:39:07,918 - INFO - Epoch 4/5, Batch 33/144 Batch Loss: 0.0125
2024-06-22 18:39:09,647 - INFO - Epoch 4/5, Batch 34/144 Batch Loss: 0.0130
2024-06-22 18:39:11,376 - INFO - Epoch 4/5, Batch 35/144 Batch Loss: 0.0129
2024-06-22 18:39:13,122 - INFO - Epoch 4/5, Batch 36/144 Batch Loss: 0.0127
2024-06-22 18:39:14,855 - INFO - Epoch 4/5, Batch 37/144 Batch Loss: 0.0131
2024-06-22 18:39:16,598 - INFO - Epoch 4/5, Batch 38/144 Batch Loss: 0.0129
2024-06-22 18:39:18,328 - INFO - Epoch 4/5, Batch 39/144 Batch Loss: 0.0121
2024-06-22 18:39:20,073 - INFO - Epoch 4/5, Batch 40/144 Batch Loss: 0.0133
2024-06-22 18:39:21,801 - INFO - Epoch 4/5, Batch 41/144 Batch Loss: 0.0129
2024-06-22 18:39:23,533 - INFO - Epoch 4/5, Batch 42/144 Batch Loss: 0.0131
2024-06-22 18:39:25,260 - INFO - Epoch 4/5, Batch 43/144 Batch Loss: 0.0135
2024-06-22 18:39:26,994 - INFO - Epoch 4/5, Batch 44/144 Batch Loss: 0.0126
2024-06-22 18:39:28,797 - INFO - Epoch 4/5, Batch 45/144 Batch Loss: 0.0128
2024-06-22 18:39:30,556 - INFO - Epoch 4/5, Batch 46/144 Batch Loss: 0.0126
2024-06-22 18:39:32,303 - INFO - Epoch 4/5, Batch 47/144 Batch Loss: 0.0137
2024-06-22 18:39:34,033 - INFO - Epoch 4/5, Batch 48/144 Batch Loss: 0.0125
2024-06-22 18:39:35,772 - INFO - Epoch 4/5, Batch 49/144 Batch Loss: 0.0140
2024-06-22 18:39:37,501 - INFO - Epoch 4/5, Batch 50/144 Batch Loss: 0.0125
2024-06-22 18:39:39,268 - INFO - Epoch 4/5, Batch 51/144 Batch Loss: 0.0129
2024-06-22 18:39:41,013 - INFO - Epoch 4/5, Batch 52/144 Batch Loss: 0.0117
2024-06-22 18:39:42,746 - INFO - Epoch 4/5, Batch 53/144 Batch Loss: 0.0126
2024-06-22 18:39:44,486 - INFO - Epoch 4/5, Batch 54/144 Batch Loss: 0.0124
2024-06-22 18:39:46,214 - INFO - Epoch 4/5, Batch 55/144 Batch Loss: 0.0142
2024-06-22 18:39:47,974 - INFO - Epoch 4/5, Batch 56/144 Batch Loss: 0.0121
2024-06-22 18:39:49,563 - INFO - Epoch 4/5, Batch 57/144 Batch Loss: 0.0125
2024-06-22 18:39:51,305 - INFO - Epoch 4/5, Batch 58/144 Batch Loss: 0.0137
2024-06-22 18:39:53,041 - INFO - Epoch 4/5, Batch 59/144 Batch Loss: 0.0126
2024-06-22 18:39:54,770 - INFO - Epoch 4/5, Batch 60/144 Batch Loss: 0.0144
2024-06-22 18:39:56,510 - INFO - Epoch 4/5, Batch 61/144 Batch Loss: 0.0129
2024-06-22 18:39:58,241 - INFO - Epoch 4/5, Batch 62/144 Batch Loss: 0.0122
2024-06-22 18:39:59,986 - INFO - Epoch 4/5, Batch 63/144 Batch Loss: 0.0132
2024-06-22 18:40:01,756 - INFO - Epoch 4/5, Batch 64/144 Batch Loss: 0.0128
2024-06-22 18:40:03,487 - INFO - Epoch 4/5, Batch 65/144 Batch Loss: 0.0119
2024-06-22 18:40:05,232 - INFO - Epoch 4/5, Batch 66/144 Batch Loss: 0.0138
2024-06-22 18:40:06,961 - INFO - Epoch 4/5, Batch 67/144 Batch Loss: 0.0123
2024-06-22 18:40:08,705 - INFO - Epoch 4/5, Batch 68/144 Batch Loss: 0.0123
2024-06-22 18:40:10,436 - INFO - Epoch 4/5, Batch 69/144 Batch Loss: 0.0139
2024-06-22 18:40:12,186 - INFO - Epoch 4/5, Batch 70/144 Batch Loss: 0.0132
2024-06-22 18:40:13,955 - INFO - Epoch 4/5, Batch 71/144 Batch Loss: 0.0127
2024-06-22 18:40:15,704 - INFO - Epoch 4/5, Batch 72/144 Batch Loss: 0.0134
2024-06-22 18:40:17,448 - INFO - Epoch 4/5, Batch 73/144 Batch Loss: 0.0134
2024-06-22 18:40:19,193 - INFO - Epoch 4/5, Batch 74/144 Batch Loss: 0.0121
2024-06-22 18:40:20,924 - INFO - Epoch 4/5, Batch 75/144 Batch Loss: 0.0123
2024-06-22 18:40:22,666 - INFO - Epoch 4/5, Batch 76/144 Batch Loss: 0.0131
2024-06-22 18:40:24,397 - INFO - Epoch 4/5, Batch 77/144 Batch Loss: 0.0127
2024-06-22 18:40:26,143 - INFO - Epoch 4/5, Batch 78/144 Batch Loss: 0.0135
2024-06-22 18:40:27,911 - INFO - Epoch 4/5, Batch 79/144 Batch Loss: 0.0137
2024-06-22 18:40:29,656 - INFO - Epoch 4/5, Batch 80/144 Batch Loss: 0.0120
2024-06-22 18:40:31,389 - INFO - Epoch 4/5, Batch 81/144 Batch Loss: 0.0127
2024-06-22 18:40:33,118 - INFO - Epoch 4/5, Batch 82/144 Batch Loss: 0.0132
2024-06-22 18:40:34,864 - INFO - Epoch 4/5, Batch 83/144 Batch Loss: 0.0126
2024-06-22 18:40:36,591 - INFO - Epoch 4/5, Batch 84/144 Batch Loss: 0.0134
2024-06-22 18:40:38,323 - INFO - Epoch 4/5, Batch 85/144 Batch Loss: 0.0116
2024-06-22 18:40:40,070 - INFO - Epoch 4/5, Batch 86/144 Batch Loss: 0.0136
2024-06-22 18:40:41,798 - INFO - Epoch 4/5, Batch 87/144 Batch Loss: 0.0129
2024-06-22 18:40:43,543 - INFO - Epoch 4/5, Batch 88/144 Batch Loss: 0.0122
2024-06-22 18:40:45,309 - INFO - Epoch 4/5, Batch 89/144 Batch Loss: 0.0128
2024-06-22 18:40:47,057 - INFO - Epoch 4/5, Batch 90/144 Batch Loss: 0.0126
2024-06-22 18:40:48,789 - INFO - Epoch 4/5, Batch 91/144 Batch Loss: 0.0124
2024-06-22 18:40:50,517 - INFO - Epoch 4/5, Batch 92/144 Batch Loss: 0.0130
2024-06-22 18:40:52,259 - INFO - Epoch 4/5, Batch 93/144 Batch Loss: 0.0126
2024-06-22 18:40:53,992 - INFO - Epoch 4/5, Batch 94/144 Batch Loss: 0.0132
2024-06-22 18:40:55,736 - INFO - Epoch 4/5, Batch 95/144 Batch Loss: 0.0127
2024-06-22 18:40:57,466 - INFO - Epoch 4/5, Batch 96/144 Batch Loss: 0.0131
2024-06-22 18:40:59,210 - INFO - Epoch 4/5, Batch 97/144 Batch Loss: 0.0128
2024-06-22 18:41:00,941 - INFO - Epoch 4/5, Batch 98/144 Batch Loss: 0.0124
2024-06-22 18:41:02,692 - INFO - Epoch 4/5, Batch 99/144 Batch Loss: 0.0131
2024-06-22 18:41:04,419 - INFO - Epoch 4/5, Batch 100/144 Batch Loss: 0.0131
2024-06-22 18:41:06,200 - INFO - Epoch 4/5, Batch 101/144 Batch Loss: 0.0115
2024-06-22 18:41:07,928 - INFO - Epoch 4/5, Batch 102/144 Batch Loss: 0.0130
2024-06-22 18:41:09,676 - INFO - Epoch 4/5, Batch 103/144 Batch Loss: 0.0122
2024-06-22 18:41:11,406 - INFO - Epoch 4/5, Batch 104/144 Batch Loss: 0.0129
2024-06-22 18:41:13,138 - INFO - Epoch 4/5, Batch 105/144 Batch Loss: 0.0128
2024-06-22 18:41:14,871 - INFO - Epoch 4/5, Batch 106/144 Batch Loss: 0.0135
2024-06-22 18:41:16,614 - INFO - Epoch 4/5, Batch 107/144 Batch Loss: 0.0130
2024-06-22 18:41:18,343 - INFO - Epoch 4/5, Batch 108/144 Batch Loss: 0.0132
2024-06-22 18:41:20,075 - INFO - Epoch 4/5, Batch 109/144 Batch Loss: 0.0123
2024-06-22 18:41:21,825 - INFO - Epoch 4/5, Batch 110/144 Batch Loss: 0.0126
2024-06-22 18:41:23,554 - INFO - Epoch 4/5, Batch 111/144 Batch Loss: 0.0123
2024-06-22 18:41:25,283 - INFO - Epoch 4/5, Batch 112/144 Batch Loss: 0.0116
2024-06-22 18:41:27,029 - INFO - Epoch 4/5, Batch 113/144 Batch Loss: 0.0115
2024-06-22 18:41:28,761 - INFO - Epoch 4/5, Batch 114/144 Batch Loss: 0.0125
2024-06-22 18:41:30,509 - INFO - Epoch 4/5, Batch 115/144 Batch Loss: 0.0120
2024-06-22 18:41:32,269 - INFO - Epoch 4/5, Batch 116/144 Batch Loss: 0.0125
2024-06-22 18:41:34,003 - INFO - Epoch 4/5, Batch 117/144 Batch Loss: 0.0126
2024-06-22 18:41:35,749 - INFO - Epoch 4/5, Batch 118/144 Batch Loss: 0.0126
2024-06-22 18:41:37,479 - INFO - Epoch 4/5, Batch 119/144 Batch Loss: 0.0126
2024-06-22 18:41:39,229 - INFO - Epoch 4/5, Batch 120/144 Batch Loss: 0.0131
2024-06-22 18:41:40,961 - INFO - Epoch 4/5, Batch 121/144 Batch Loss: 0.0125
2024-06-22 18:41:42,705 - INFO - Epoch 4/5, Batch 122/144 Batch Loss: 0.0117
2024-06-22 18:41:44,435 - INFO - Epoch 4/5, Batch 123/144 Batch Loss: 0.0130
2024-06-22 18:41:46,178 - INFO - Epoch 4/5, Batch 124/144 Batch Loss: 0.0128
2024-06-22 18:41:47,912 - INFO - Epoch 4/5, Batch 125/144 Batch Loss: 0.0127
2024-06-22 18:41:49,657 - INFO - Epoch 4/5, Batch 126/144 Batch Loss: 0.0130
2024-06-22 18:41:51,382 - INFO - Epoch 4/5, Batch 127/144 Batch Loss: 0.0124
2024-06-22 18:41:53,112 - INFO - Epoch 4/5, Batch 128/144 Batch Loss: 0.0129
2024-06-22 18:41:54,846 - INFO - Epoch 4/5, Batch 129/144 Batch Loss: 0.0122
2024-06-22 18:41:56,589 - INFO - Epoch 4/5, Batch 130/144 Batch Loss: 0.0121
2024-06-22 18:41:58,319 - INFO - Epoch 4/5, Batch 131/144 Batch Loss: 0.0124
2024-06-22 18:42:00,054 - INFO - Epoch 4/5, Batch 132/144 Batch Loss: 0.0124
2024-06-22 18:42:01,829 - INFO - Epoch 4/5, Batch 133/144 Batch Loss: 0.0119
2024-06-22 18:42:03,562 - INFO - Epoch 4/5, Batch 134/144 Batch Loss: 0.0119
2024-06-22 18:42:05,306 - INFO - Epoch 4/5, Batch 135/144 Batch Loss: 0.0126
2024-06-22 18:42:07,039 - INFO - Epoch 4/5, Batch 136/144 Batch Loss: 0.0137
2024-06-22 18:42:08,785 - INFO - Epoch 4/5, Batch 137/144 Batch Loss: 0.0125
2024-06-22 18:42:10,537 - INFO - Epoch 4/5, Batch 138/144 Batch Loss: 0.0129
2024-06-22 18:42:12,287 - INFO - Epoch 4/5, Batch 139/144 Batch Loss: 0.0122
2024-06-22 18:42:14,018 - INFO - Epoch 4/5, Batch 140/144 Batch Loss: 0.0124
2024-06-22 18:42:15,766 - INFO - Epoch 4/5, Batch 141/144 Batch Loss: 0.0123
2024-06-22 18:42:17,493 - INFO - Epoch 4/5, Batch 142/144 Batch Loss: 0.0119
2024-06-22 18:42:19,225 - INFO - Epoch 4/5, Batch 143/144 Batch Loss: 0.0128
2024-06-22 18:42:20,973 - INFO - Epoch 4/5, Batch 144/144 Batch Loss: 0.0126
2024-06-22 18:42:58,984 - INFO - Epoch 4/5 - Train Loss: 0.0127, Valid Loss: 0.0182, 
2024-06-22 18:43:00,720 - INFO - Epoch 5/5, Batch 1/144 Batch Loss: 0.0121
2024-06-22 18:43:02,455 - INFO - Epoch 5/5, Batch 2/144 Batch Loss: 0.0126
2024-06-22 18:43:04,197 - INFO - Epoch 5/5, Batch 3/144 Batch Loss: 0.0126
2024-06-22 18:43:05,930 - INFO - Epoch 5/5, Batch 4/144 Batch Loss: 0.0122
2024-06-22 18:43:07,665 - INFO - Epoch 5/5, Batch 5/144 Batch Loss: 0.0137
2024-06-22 18:43:09,410 - INFO - Epoch 5/5, Batch 6/144 Batch Loss: 0.0126
2024-06-22 18:43:11,144 - INFO - Epoch 5/5, Batch 7/144 Batch Loss: 0.0133
2024-06-22 18:43:12,895 - INFO - Epoch 5/5, Batch 8/144 Batch Loss: 0.0123
2024-06-22 18:43:14,626 - INFO - Epoch 5/5, Batch 9/144 Batch Loss: 0.0129
2024-06-22 18:43:16,374 - INFO - Epoch 5/5, Batch 10/144 Batch Loss: 0.0123
2024-06-22 18:43:18,108 - INFO - Epoch 5/5, Batch 11/144 Batch Loss: 0.0124
2024-06-22 18:43:19,852 - INFO - Epoch 5/5, Batch 12/144 Batch Loss: 0.0130
2024-06-22 18:43:21,618 - INFO - Epoch 5/5, Batch 13/144 Batch Loss: 0.0125
2024-06-22 18:43:23,350 - INFO - Epoch 5/5, Batch 14/144 Batch Loss: 0.0125
2024-06-22 18:43:25,079 - INFO - Epoch 5/5, Batch 15/144 Batch Loss: 0.0125
2024-06-22 18:43:26,812 - INFO - Epoch 5/5, Batch 16/144 Batch Loss: 0.0132
2024-06-22 18:43:28,562 - INFO - Epoch 5/5, Batch 17/144 Batch Loss: 0.0127
2024-06-22 18:43:30,294 - INFO - Epoch 5/5, Batch 18/144 Batch Loss: 0.0143
2024-06-22 18:43:32,087 - INFO - Epoch 5/5, Batch 19/144 Batch Loss: 0.0137
2024-06-22 18:43:33,826 - INFO - Epoch 5/5, Batch 20/144 Batch Loss: 0.0122
2024-06-22 18:43:35,555 - INFO - Epoch 5/5, Batch 21/144 Batch Loss: 0.0129
2024-06-22 18:43:37,287 - INFO - Epoch 5/5, Batch 22/144 Batch Loss: 0.0128
2024-06-22 18:43:39,021 - INFO - Epoch 5/5, Batch 23/144 Batch Loss: 0.0119
2024-06-22 18:43:40,751 - INFO - Epoch 5/5, Batch 24/144 Batch Loss: 0.0129
2024-06-22 18:43:42,484 - INFO - Epoch 5/5, Batch 25/144 Batch Loss: 0.0128
2024-06-22 18:43:44,223 - INFO - Epoch 5/5, Batch 26/144 Batch Loss: 0.0132
2024-06-22 18:43:45,993 - INFO - Epoch 5/5, Batch 27/144 Batch Loss: 0.0128
2024-06-22 18:43:47,735 - INFO - Epoch 5/5, Batch 28/144 Batch Loss: 0.0130
2024-06-22 18:43:49,469 - INFO - Epoch 5/5, Batch 29/144 Batch Loss: 0.0129
2024-06-22 18:43:51,202 - INFO - Epoch 5/5, Batch 30/144 Batch Loss: 0.0128
2024-06-22 18:43:52,842 - INFO - Epoch 5/5, Batch 31/144 Batch Loss: 0.0135
2024-06-22 18:43:54,557 - INFO - Epoch 5/5, Batch 32/144 Batch Loss: 0.0118
2024-06-22 18:43:56,289 - INFO - Epoch 5/5, Batch 33/144 Batch Loss: 0.0119
2024-06-22 18:43:58,019 - INFO - Epoch 5/5, Batch 34/144 Batch Loss: 0.0126
2024-06-22 18:43:59,792 - INFO - Epoch 5/5, Batch 35/144 Batch Loss: 0.0120
2024-06-22 18:44:01,525 - INFO - Epoch 5/5, Batch 36/144 Batch Loss: 0.0122
2024-06-22 18:44:03,096 - INFO - Epoch 5/5, Batch 37/144 Batch Loss: 0.0114
2024-06-22 18:44:04,790 - INFO - Epoch 5/5, Batch 38/144 Batch Loss: 0.0125
2024-06-22 18:44:06,516 - INFO - Epoch 5/5, Batch 39/144 Batch Loss: 0.0124
2024-06-22 18:44:08,240 - INFO - Epoch 5/5, Batch 40/144 Batch Loss: 0.0127
2024-06-22 18:44:09,998 - INFO - Epoch 5/5, Batch 41/144 Batch Loss: 0.0117
2024-06-22 18:44:11,721 - INFO - Epoch 5/5, Batch 42/144 Batch Loss: 0.0124
2024-06-22 18:44:13,463 - INFO - Epoch 5/5, Batch 43/144 Batch Loss: 0.0135
2024-06-22 18:44:15,181 - INFO - Epoch 5/5, Batch 44/144 Batch Loss: 0.0117
2024-06-22 18:44:16,887 - INFO - Epoch 5/5, Batch 45/144 Batch Loss: 0.0128
2024-06-22 18:44:18,579 - INFO - Epoch 5/5, Batch 46/144 Batch Loss: 0.0136
2024-06-22 18:44:20,311 - INFO - Epoch 5/5, Batch 47/144 Batch Loss: 0.0118
2024-06-22 18:44:22,040 - INFO - Epoch 5/5, Batch 48/144 Batch Loss: 0.0126
2024-06-22 18:44:23,736 - INFO - Epoch 5/5, Batch 49/144 Batch Loss: 0.0128
2024-06-22 18:44:25,427 - INFO - Epoch 5/5, Batch 50/144 Batch Loss: 0.0132
2024-06-22 18:44:27,117 - INFO - Epoch 5/5, Batch 51/144 Batch Loss: 0.0131
2024-06-22 18:44:28,808 - INFO - Epoch 5/5, Batch 52/144 Batch Loss: 0.0136
2024-06-22 18:44:30,497 - INFO - Epoch 5/5, Batch 53/144 Batch Loss: 0.0118
2024-06-22 18:44:32,229 - INFO - Epoch 5/5, Batch 54/144 Batch Loss: 0.0127
2024-06-22 18:44:33,930 - INFO - Epoch 5/5, Batch 55/144 Batch Loss: 0.0121
2024-06-22 18:44:35,615 - INFO - Epoch 5/5, Batch 56/144 Batch Loss: 0.0125
2024-06-22 18:44:37,281 - INFO - Epoch 5/5, Batch 57/144 Batch Loss: 0.0121
2024-06-22 18:44:38,964 - INFO - Epoch 5/5, Batch 58/144 Batch Loss: 0.0137
2024-06-22 18:44:40,626 - INFO - Epoch 5/5, Batch 59/144 Batch Loss: 0.0134
2024-06-22 18:44:42,312 - INFO - Epoch 5/5, Batch 60/144 Batch Loss: 0.0135
2024-06-22 18:44:43,978 - INFO - Epoch 5/5, Batch 61/144 Batch Loss: 0.0121
2024-06-22 18:44:45,657 - INFO - Epoch 5/5, Batch 62/144 Batch Loss: 0.0130
2024-06-22 18:44:47,354 - INFO - Epoch 5/5, Batch 63/144 Batch Loss: 0.0136
2024-06-22 18:44:49,060 - INFO - Epoch 5/5, Batch 64/144 Batch Loss: 0.0126
2024-06-22 18:44:50,725 - INFO - Epoch 5/5, Batch 65/144 Batch Loss: 0.0127
2024-06-22 18:44:52,425 - INFO - Epoch 5/5, Batch 66/144 Batch Loss: 0.0127
2024-06-22 18:44:54,087 - INFO - Epoch 5/5, Batch 67/144 Batch Loss: 0.0131
2024-06-22 18:44:55,770 - INFO - Epoch 5/5, Batch 68/144 Batch Loss: 0.0127
2024-06-22 18:44:57,433 - INFO - Epoch 5/5, Batch 69/144 Batch Loss: 0.0130
2024-06-22 18:44:59,121 - INFO - Epoch 5/5, Batch 70/144 Batch Loss: 0.0128
2024-06-22 18:45:00,816 - INFO - Epoch 5/5, Batch 71/144 Batch Loss: 0.0126
2024-06-22 18:45:02,493 - INFO - Epoch 5/5, Batch 72/144 Batch Loss: 0.0115
2024-06-22 18:45:04,158 - INFO - Epoch 5/5, Batch 73/144 Batch Loss: 0.0122
2024-06-22 18:45:05,833 - INFO - Epoch 5/5, Batch 74/144 Batch Loss: 0.0126
2024-06-22 18:45:07,493 - INFO - Epoch 5/5, Batch 75/144 Batch Loss: 0.0123
2024-06-22 18:45:09,173 - INFO - Epoch 5/5, Batch 76/144 Batch Loss: 0.0122
2024-06-22 18:45:10,906 - INFO - Epoch 5/5, Batch 77/144 Batch Loss: 0.0139
2024-06-22 18:45:12,585 - INFO - Epoch 5/5, Batch 78/144 Batch Loss: 0.0123
2024-06-22 18:45:14,245 - INFO - Epoch 5/5, Batch 79/144 Batch Loss: 0.0131
2024-06-22 18:45:15,949 - INFO - Epoch 5/5, Batch 80/144 Batch Loss: 0.0124
2024-06-22 18:45:17,615 - INFO - Epoch 5/5, Batch 81/144 Batch Loss: 0.0125
2024-06-22 18:45:19,298 - INFO - Epoch 5/5, Batch 82/144 Batch Loss: 0.0133
2024-06-22 18:45:20,961 - INFO - Epoch 5/5, Batch 83/144 Batch Loss: 0.0118
2024-06-22 18:45:22,679 - INFO - Epoch 5/5, Batch 84/144 Batch Loss: 0.0129
2024-06-22 18:45:24,358 - INFO - Epoch 5/5, Batch 85/144 Batch Loss: 0.0122
2024-06-22 18:45:26,041 - INFO - Epoch 5/5, Batch 86/144 Batch Loss: 0.0124
2024-06-22 18:45:27,706 - INFO - Epoch 5/5, Batch 87/144 Batch Loss: 0.0120
2024-06-22 18:45:29,389 - INFO - Epoch 5/5, Batch 88/144 Batch Loss: 0.0122
2024-06-22 18:45:31,053 - INFO - Epoch 5/5, Batch 89/144 Batch Loss: 0.0124
2024-06-22 18:45:32,739 - INFO - Epoch 5/5, Batch 90/144 Batch Loss: 0.0128
2024-06-22 18:45:34,402 - INFO - Epoch 5/5, Batch 91/144 Batch Loss: 0.0117
2024-06-22 18:45:36,098 - INFO - Epoch 5/5, Batch 92/144 Batch Loss: 0.0137
2024-06-22 18:45:37,779 - INFO - Epoch 5/5, Batch 93/144 Batch Loss: 0.0128
2024-06-22 18:45:39,460 - INFO - Epoch 5/5, Batch 94/144 Batch Loss: 0.0121
2024-06-22 18:45:41,123 - INFO - Epoch 5/5, Batch 95/144 Batch Loss: 0.0131
2024-06-22 18:45:42,806 - INFO - Epoch 5/5, Batch 96/144 Batch Loss: 0.0132
2024-06-22 18:45:44,468 - INFO - Epoch 5/5, Batch 97/144 Batch Loss: 0.0125
2024-06-22 18:45:46,148 - INFO - Epoch 5/5, Batch 98/144 Batch Loss: 0.0129
2024-06-22 18:45:47,808 - INFO - Epoch 5/5, Batch 99/144 Batch Loss: 0.0123
2024-06-22 18:45:49,487 - INFO - Epoch 5/5, Batch 100/144 Batch Loss: 0.0127
2024-06-22 18:45:51,152 - INFO - Epoch 5/5, Batch 101/144 Batch Loss: 0.0128
2024-06-22 18:45:52,833 - INFO - Epoch 5/5, Batch 102/144 Batch Loss: 0.0134
2024-06-22 18:45:54,540 - INFO - Epoch 5/5, Batch 103/144 Batch Loss: 0.0123
2024-06-22 18:45:56,222 - INFO - Epoch 5/5, Batch 104/144 Batch Loss: 0.0135
2024-06-22 18:45:57,882 - INFO - Epoch 5/5, Batch 105/144 Batch Loss: 0.0137
2024-06-22 18:45:59,584 - INFO - Epoch 5/5, Batch 106/144 Batch Loss: 0.0113
2024-06-22 18:46:01,241 - INFO - Epoch 5/5, Batch 107/144 Batch Loss: 0.0113
2024-06-22 18:46:02,953 - INFO - Epoch 5/5, Batch 108/144 Batch Loss: 0.0128
2024-06-22 18:46:04,615 - INFO - Epoch 5/5, Batch 109/144 Batch Loss: 0.0124
2024-06-22 18:46:06,295 - INFO - Epoch 5/5, Batch 110/144 Batch Loss: 0.0123
2024-06-22 18:46:07,957 - INFO - Epoch 5/5, Batch 111/144 Batch Loss: 0.0120
2024-06-22 18:46:09,636 - INFO - Epoch 5/5, Batch 112/144 Batch Loss: 0.0128
2024-06-22 18:46:11,296 - INFO - Epoch 5/5, Batch 113/144 Batch Loss: 0.0118
2024-06-22 18:46:12,981 - INFO - Epoch 5/5, Batch 114/144 Batch Loss: 0.0120
2024-06-22 18:46:14,680 - INFO - Epoch 5/5, Batch 115/144 Batch Loss: 0.0121
2024-06-22 18:46:16,363 - INFO - Epoch 5/5, Batch 116/144 Batch Loss: 0.0123
2024-06-22 18:46:18,024 - INFO - Epoch 5/5, Batch 117/144 Batch Loss: 0.0119
2024-06-22 18:46:19,706 - INFO - Epoch 5/5, Batch 118/144 Batch Loss: 0.0121
2024-06-22 18:46:21,363 - INFO - Epoch 5/5, Batch 119/144 Batch Loss: 0.0123
2024-06-22 18:46:23,061 - INFO - Epoch 5/5, Batch 120/144 Batch Loss: 0.0137
2024-06-22 18:46:24,726 - INFO - Epoch 5/5, Batch 121/144 Batch Loss: 0.0123
2024-06-22 18:46:26,408 - INFO - Epoch 5/5, Batch 122/144 Batch Loss: 0.0118
2024-06-22 18:46:28,073 - INFO - Epoch 5/5, Batch 123/144 Batch Loss: 0.0129
2024-06-22 18:46:29,757 - INFO - Epoch 5/5, Batch 124/144 Batch Loss: 0.0124
2024-06-22 18:46:31,418 - INFO - Epoch 5/5, Batch 125/144 Batch Loss: 0.0125
2024-06-22 18:46:33,099 - INFO - Epoch 5/5, Batch 126/144 Batch Loss: 0.0130
2024-06-22 18:46:34,761 - INFO - Epoch 5/5, Batch 127/144 Batch Loss: 0.0122
2024-06-22 18:46:36,474 - INFO - Epoch 5/5, Batch 128/144 Batch Loss: 0.0116
2024-06-22 18:46:38,171 - INFO - Epoch 5/5, Batch 129/144 Batch Loss: 0.0122
2024-06-22 18:46:39,858 - INFO - Epoch 5/5, Batch 130/144 Batch Loss: 0.0124
2024-06-22 18:46:41,516 - INFO - Epoch 5/5, Batch 131/144 Batch Loss: 0.0124
2024-06-22 18:46:43,201 - INFO - Epoch 5/5, Batch 132/144 Batch Loss: 0.0110
2024-06-22 18:46:44,858 - INFO - Epoch 5/5, Batch 133/144 Batch Loss: 0.0130
2024-06-22 18:46:46,539 - INFO - Epoch 5/5, Batch 134/144 Batch Loss: 0.0124
2024-06-22 18:46:48,234 - INFO - Epoch 5/5, Batch 135/144 Batch Loss: 0.0126
2024-06-22 18:46:49,939 - INFO - Epoch 5/5, Batch 136/144 Batch Loss: 0.0131
2024-06-22 18:46:51,601 - INFO - Epoch 5/5, Batch 137/144 Batch Loss: 0.0129
2024-06-22 18:46:53,283 - INFO - Epoch 5/5, Batch 138/144 Batch Loss: 0.0126
2024-06-22 18:46:54,970 - INFO - Epoch 5/5, Batch 139/144 Batch Loss: 0.0122
2024-06-22 18:46:56,651 - INFO - Epoch 5/5, Batch 140/144 Batch Loss: 0.0118
2024-06-22 18:46:58,315 - INFO - Epoch 5/5, Batch 141/144 Batch Loss: 0.0122
2024-06-22 18:47:00,013 - INFO - Epoch 5/5, Batch 142/144 Batch Loss: 0.0118
2024-06-22 18:47:01,695 - INFO - Epoch 5/5, Batch 143/144 Batch Loss: 0.0122
2024-06-22 18:47:03,375 - INFO - Epoch 5/5, Batch 144/144 Batch Loss: 0.0136
2024-06-22 18:47:39,084 - INFO - Epoch 5/5 - Train Loss: 0.0125, Valid Loss: 0.0181, 
2024-06-22 18:47:47,556 - INFO - Test Loss: 0.0185
2024-06-22 18:47:47,556 - INFO - Elapsed time -> 22.9880 minutes
