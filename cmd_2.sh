#!/bin/bash
# intx=0 
# start_x=267
# start_y=467
# interval=66
# while(($intx <=1))
#     do
#         inty=0
#         posx=$(expr $start_x + $interval \* $intx)
#         while(($inty<=1))
#             do 
#                 posy=$(expr $start_y + $interval \* $inty)
#                 python ~/JIDI_Competition/rl_trainer/main.py --goalx $posx --goaly $posy --cpu 6 --epoch_step 600 --train_epoch 100
#                 let "inty++"
#             done
#         let "intx++"
#     done
python ~/JIDI_Competition/rl_trainer/main.py --goalx 267 --goaly 500 --cpu 6 --epoch_step 600 --train_epoch 100
python ~/JIDI_Competition/rl_trainer/main.py --goalx 333 --goaly 500 --cpu 6 --epoch_step 600 --train_epoch 100
python ~/JIDI_Competition/rl_trainer/main.py --goalx 300 --goaly 467 --cpu 6 --epoch_step 600 --train_epoch 100
python ~/JIDI_Competition/rl_trainer/main.py --goalx 300 --goaly 533 --cpu 6 --epoch_step 600 --train_epoch 100
