dataset=/tungstenfs/scratch/shared/gchao_ggiorget/datasets/publication
basedir=/tungstenfs/scratch/shared/gchao_ggiorget/benchmarks/trackmate

nice -n 19 python trackmate_prepare.py --dataset ${dataset}/microtubule.npz --output ${basedir}
nice -n 19 python trackmate_prepare.py --dataset ${dataset}/particle.npz --output ${basedir}
nice -n 19 python trackmate_prepare.py --dataset ${dataset}/receptor.npz --output ${basedir}
nice -n 19 python trackmate_prepare.py --dataset ${dataset}/smfish.npz --output ${basedir}
nice -n 19 python trackmate_prepare.py --dataset ${dataset}/suntag.npz --output ${basedir}
nice -n 19 python trackmate_prepare.py --dataset ${dataset}/vesicle.npz --output ${basedir}

nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="${basedir}/microtubule"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="${basedir}/particle"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="${basedir}/receptor"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="${basedir}/smfish"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="${basedir}/suntag"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="${basedir}/vesicle"'

nice -n 19 python trackmate_train.py --basedir ${basedir}/microtubule
nice -n 19 python trackmate_train.py --basedir ${basedir}/particle
nice -n 19 python trackmate_train.py --basedir ${basedir}/receptor
nice -n 19 python trackmate_train.py --basedir ${basedir}/smfish
nice -n 19 python trackmate_train.py --basedir ${basedir}/suntag
nice -n 19 python trackmate_train.py --basedir ${basedir}/vesicle

nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py 'basedir="${basedir}/microtubule",detector="log",radius="3.0",median="False"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py 'basedir="${basedir}/particle",detector="log",radius="3.0",median="False"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py 'basedir="${basedir}/receptor",detector="log",radius="3.0",median="False"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py 'basedir="${basedir}/smfish",detector="log",radius="5.0",median="False"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py 'basedir="${basedir}/suntag",detector="log",radius="1.0",median="False"'
nice -n 19 Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py 'basedir="${basedir}/vesicle",detector="log",radius="5.0",median="False"'

nice -n 19 python trackmate_test.py --basedir ${basedir}/microtubule --threshold 1.091255
nice -n 19 python trackmate_test.py --basedir ${basedir}/particle --threshold 0.021363
nice -n 19 python trackmate_test.py --basedir ${basedir}/receptor --threshold 0.794764
nice -n 19 python trackmate_test.py --basedir ${basedir}/smfish --threshold 8.071781
nice -n 19 python trackmate_test.py --basedir ${basedir}/suntag --threshold 0.170
nice -n 19 python trackmate_test.py --basedir ${basedir}/vesicle --threshold 0.0625
