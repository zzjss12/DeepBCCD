<div align="center">
  <a href="#">
  	<img src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy-downsized.gif" alt="Logo project" height="160" />
  </a>
  <br>
  <br>
  <p>
    <b>DeepBCCD</b>
  </p>
  <p>
     <i>DeepBCCD is designed to detect binary code similarities. DeepBCCD has made great improvements on the basis of the Gemini method, DeepBCCD not only uses the CFG structure information in the upper binary code, but also uses the LSTM network to extract the sequence information between the binary code instructions.</i>
  </p>
</div>

---

**Content**

* [Description](##description)
* [Install](##install)
* [Usage](##usage)
* [Exemples](##exemples)
* [Documentation](##documentation)
* [Datasets](##datasets)
* [Evaluation](##evaluation)
* [Maintainers](##maintainers)

## Description ✨
Gemini is a way to leverage structural information between basic blocks, but it doesn't take into account sequential relationships between instructions。DeepBCCD is a binary code clone detection method, which is improved on the basis of the Gemini method, and the final **AUC** value can reach about **99.7%**. For more specific information, please refer to paper------.

## Install 🐙
It is recommended that you install a conda environment and then install the dependent packages with the following command：
```
conda create -n DeepBCCD37 -y python==3.7.16 && conda activate DeepBCCD37
pip install -r requirements.txt
```

## Usage 💡
1. git clone the project.
```
git clone https:// -d your_profile
```
2. Go inside the project folder(IDE) and open your terminal.
3. See  [Install](##install) to install the environment.
4. run the command `python run.py --train true --test true` to start.

## Exemples 🖍
We trained with the **A5000**.
```
python run.py --train true --test true --w2v_dim 100 --batch_size 512--max_block_seq 20--num_block 20 --iter_level 5 
```

## Documentation 📄
For a more detailed description of the contents of DeepBCCD, please refer to our paper-----

## Datasets 👩‍💻
For the datasets, we used the datasets **BinaryCorp-3M**（https://github.com/vul337/jTrans) in the Jtrans paper. To conform to the input format according to the DeepBCCD model, We re-extracted the binary function set from the source binary and formed the **dataset_train.csv** and **dataset_test.csv** datasets，which are also essentially derived from Binarycorp-3M.
For the **BinaryCrop-26M** dataset, we will try it in the future because it requires a larger training resource。					

The dataset used in DeepBCCD 【download】(https://efss.qloud.my/index.php/s/a2B2S9rNwdXkmBo)
## Evaluation 🍰


## Maintainers 👷
* @zzjss12
* Email:s211231082@stu.cqupt.edu.cn

## License ⚖️
GPL

---
<div align="center">
	<b>
		<a href="https://www.npmjs.com/package/get-good-readme">File generated with get-good-readme module</a>
	</b>
</div>
