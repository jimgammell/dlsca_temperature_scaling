python main.py --train-gan weird_gan__tinyaes_c --device cuda:0 --cudnn-benchmark --print-to-terminal &> trial_3.txt &
python main.py --train-gan weird_gan__tinyaes_d --device cuda:1 --cudnn-benchmark --print-to-terminal &> trial_4.txt &
python main.py --train-gan weird_gan__tinyaes_e --device cuda:2 --cudnn-benchmark --print-to-terminal &> trial_5.txt &
