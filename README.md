### GCN-M

This is the implementation of GCN-M in the following paper: [Graph Convolutional Networks for Traffic Forecasting with missing values](https://arxiv.org/abs/2212.06419)



### Requirements:

- matplotlib == 3.2.1
- numpy == 1.19.2
- pandas == 0.25.1
- scikit_learn == 0.21.2
- torch == 1.6.0
- tensorwatch == 0.9.1

Dependencies can be installed using the following command:

```
pip install -r requirements.txt
```



### Data

Step1: 

- Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).
- Put the downloaded data into the repository mentioned in ***"config/DATASET.conf"***

Step2:  Preprocess raw data

```
python data/generate_dated_data.py
```



### Usage

```
python main.py --config CONFIG_FILE --itr NBR_ITERATION
```


### Citation

If you find this repository useful in your research, please consider citing the following paper:

```script
@article{zuo2022graph,
  title = {Graph Convolutional Networks for Traffic Forecasting with Missing Values},
  author = {Zuo, Jingwei and Zeitouni, Karine and Taher, Yehia and Garcia-Rodriguez, Sandra},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2212.06419},
  year = {2022}
}
```
