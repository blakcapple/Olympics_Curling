#!/bin/bash
intx=0 
start_x=200
start_y=400
interval=33
while(($intx <=6))
    do
        inty=0
        posx=$(expr $start_x + $interval \* $intx)
        while(($inty<=6))
            do 
                posy=$(expr $start_y + $interval \* $inty)
                python ~/JIDI_Competition/rl_trainer/main.py --goalx $posx --goaly $posy --cpu 6 --epoch_step 600 --train_epoch 100
                let "inty++"
            done
        let "intx++"
    done
