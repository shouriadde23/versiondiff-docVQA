# Versiondiff-docVQA
repo for versiondiff-docVQA where model receives revised versions of document, then answers comprehensive questions about changes between versions

To run each of the experiments, all of the scripts are in the scripts directory. You can use the python3 command to execute them. All the results will be stored in the results directory, in their respective experiment 1, 2, and 3 directories. 

For this project, our project extends on the initial Doc-VQA implementation.
@inproceedings{mathew2021docvqa,
  title={DocVQA: A Dataset for VQA on Document Images},
  author={Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, C. V.},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021}
}
The datasets we used are the FUNSD dataset and SROIE dataset. 

@inproceedings{jaume2019,
  title     = {FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author    = {Jaume, Guillaume and Ekenel, Hazim Kemal and Thiran, Jean-Philippe},
  booktitle = {Accepted to ICDAR-OST},
  year      = {2019}
}

This citation is for SROIE.
@article{huang2021icdar2019,
  title   = {ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
  author  = {Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and Karatzas, Dimosthenis and Lu, Shijian and Jawahar, C. V.},
  journal = {arXiv preprint arXiv:2103.10213},
  year    = {2021},
  doi     = {10.48550/arXiv.2103.10213},
  url     = {https://arxiv.org/abs/2103.10213}
}
