dataset=/tungstenfs/scratch/shared/gchao_ggiorget/datasets/publication
savedir=/tungstenfs/scratch/shared/gchao_ggiorget/benchmarks/spotlearn

nice -n 19 python detnet_train.py --dataset ${dataset}/fixed.npz --savedir ${savedir}
nice -n 19 python detnet_train.py --dataset ${dataset}/microtubule.npz --savedir ${savedir}
nice -n 19 python detnet_train.py --dataset ${dataset}/mrna.npz --savedir ${savedir}
nice -n 19 python detnet_train.py --dataset ${dataset}/receptor.npz --savedir ${savedir}
nice -n 19 python detnet_train.py --dataset ${dataset}/vesicle.npz --savedir ${savedir}

nice -n 19 python detnet_test.py --datasets ${dataset} --models ${savedir}/best
