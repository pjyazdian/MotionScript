# MotionScript: Official Implementation

## :snake: Create Python environment
<details>
<summary>Click for details.</summary>
This code was tested in a python 3.8 environment.

From the main code directory:

```bash
pip install -r requirements.txt
python setup.py develop
```
</details>

## :inbox_tray: Download and Setup

The **MotionScript** repository requires the **HumanML3D dataset**, which you can download from [HumanML3D Official](https://github.com/EricGuo5513/HumanML3D). Please follow the instructions in the original repository to set up the dataset.
If you want to include **BABEL** labels, please refer to the [official BABEL](https://babel.is.tue.mpg.de/) webpage to download the dataset

Our code is also compatible with other motion datasets, such as [Motion-X](https://github.com/IDEA-Research/Motion-X) and [BEAT](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/BEAT_2022), allowing for easy integration and application across a variety of motion capture data sources with the [SMPL-X](https://smpl-x.is.tue.mpg.de/) body representation.
### :warning: Euler angles missing in HumanML3D

The HumanML3D dataset is originally provided in **quaternions**, but converting them directly to **Euler angles** can be inefficient due to gimbal lock errors. To address this, we have modified the original [raw_pose_processing.ipynb](https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb) pipeline and now store the dataset in **Euler angles** to avoid the inefficiencies of quaternion-to-Euler conversion. You can access the [raw_pose_processing(Root-Euler).ipynb](https://colab.research.google.com/drive/1zR8M54cZY2eSC5i5j4rnPaUtcRo0ZPb6?usp=sharing) or download the processed **Euler angle** files directly from [here](https://drive.google.com/drive/folders/1ZbN85VBAGv7kRS8RQyKx-s4Hg2u-bhVV?usp=sharing).
Due to limited access, we couldn't retrieve the original [HumanAct12](https://github.com/EricGuo5513/action-to-motion) subset. For these poses, we filled missing root Euler angles with facing forward. The [AMASS](https://github.com/EricGuo5513/action-to-motion) data was successfully processed and stored in Euler angles.

### :eyes: Quick Look

To take a quick look at our generated captions and details, please refer to our [gallery](https://drive.google.com/drive/folders/1xsKcx7YbiPVx8LjFgMBNlT3VjrNYaSsz?usp=sharing). Each folder is named according to the corresponding **motion ID** in the **HumanML3D** dataset and includes the following files:

- **:movie_camera:  3DS.gif**: Animation of the body skeleton
- **:movie_camera:  Motioncodes.gif**: Detected motions displayed on a timeline
- **:movie_camera:  Merged.gif**: A side-by-side merge of the above two animations
- **:memo: sanity_check.txt**: A sanity check of the detected motions, along with additional details
- **:memo: MS_only.txt**: Generated captions by the MotionScript algorithm
- **:memo: combined.txt**: A combination of generated captions and original captions from HumanML3D as an augmentation method

## :page_with_curl: Generate automatic captions

To generate captions using the MotionScript framework, follow these steps:

- **Process the HumanML3D dataset and extract joint data**
  Update the paths to point to your data location. Once set, run the following command to extract the corresponding BABEL labels in HumanML3D based on their AMASS crops:
  ```
  python format_babel_labels.py
  ```
- **Generate captions**
  ```
  python captioning_motion.py --generate
  ```
- **Analyze motion statistics**

  Analyze motion statistics to refine thresholds and eligibility:
  ```
  python captioning_motion.py --generate
  ```
- In following file, The data structures are extensivey explained and you can customize categories, thresholds, tolerable noise levels, eligibility, and caption templates. Look for markers such as `ADD_NEW_MOTION_RULE` or `UPDATE_TEMPLATE` , etc.
  - captioning_data.py
  - posecodes.py


## Citation

If you use this code or the MotionScript dataset, please cite the following paper:

```
@article{yazdian2023motionscript,
  title={Motionscript: Natural language descriptions for expressive 3d human motions},
  author={Yazdian, Payam Jome and Liu, Eric and Lagasse, Rachel and Mohammadi, Hamid and Cheng, Li and Lim, Angelica},
  journal={arXiv preprint arXiv:2312.12634},
  year={2023}
}

```

## License

This code is distributed under an [MIT LICENSE](LICENSE).


Please feel free to contact us (pjomeyaz@sfu.ca) with any question or concerns.

