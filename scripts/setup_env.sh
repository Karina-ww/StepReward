set -x

eval "$(conda shell.bash hook)"

conda create -n viper python==3.10 -y
conda activate viper
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip install flash-attn==2.7.4.post1