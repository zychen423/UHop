# UHop: An Unrestricted-Hop Relation Extraction Framework for Knowledge-Based Question Answering 

This is the code for our UHop [paper](https://arxiv.org/abs/1904.01246). 

## Getting Started

To train and evaluate this code:
```shell
git clone https://github.com/zychen423/UHop.git
cd UHop
wget http://doraemon.iis.sinica.edu.tw/UHop/UHop_data.zip
unzip UHop_data.zip
mkdir saved_model
cd ./script
CUDA_VISIBLE_DEVICES=0 python3.6 train.py
```

Then you can use scripts in script/allocate to get best result from parameter search

### Prerequisites

The environment we use is listed:

python >= 3.6
pytorch 1.3


## License

This Work is licensed under the GNU General Public License v3.0 without any warranties. The license text in full can be getting access at the file named COPYING-GPL-3.0. Any person obtaining a copy of this Work and associated documentation files is granted the rights to use, copy, modify, merge, publish, and distribute the Work for any purpose. However if any work is based upon this Work and hence constitutes a Derivative Work, the GPL-3.0 license requires distributions of the Work and the Derivative Work to remain under the same license or a similar license with the Source Code provision obligation.


