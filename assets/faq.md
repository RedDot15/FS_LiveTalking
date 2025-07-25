1.  PyTorch3D installation failed
    Solution: Download the source code and compile it.

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
python setup.py install
```

2.  Websocket connection error
    Solution: Modify python/site-packages/flask\_sockets.py

```python
self.url_map.add(Rule(rule, endpoint=f)) # Original line in flask_sockets.py.
# Change to:
self.url_map.add(Rule(rule, endpoint=f, websocket=True))
```

3. Protobuf version too high

```bash
# Uninstalls the currently installed protobuf package.
pip uninstall protobuf 
# Installs a specific, older version of protobuf (3.20.1) that is compatible.
pip install protobuf==3.20.1 
```

4. Digital human not blinking
Solution: Add the following steps during model training:

# This refers to Action Unit 45, which corresponds to eye blinking in facial action coding systems.
> Obtain AU45 for eyes blinking.\ 

# Instructions to extract facial action unit features using OpenFace and place the resulting CSV file in the specified data directory.
> Run FeatureExtraction in OpenFace, rename and move the output CSV file to data/\<ID>/au.csv. 

# This is a direct instruction to ensure the extracted AU data is in the correct place for the project to use.
Copy au.csv to the data directory of this project. 

5. Adding a background image to the digital human

```bash
# Runs the 'app.py' script and passes '--bg_img bc.jpg' as an argument, setting 'bc.jpg' as the background image.
python app.py --bg_img bc.jpg
```

6. Model trained by yourself reports dimension mismatch errors
Solution: Use wav2vec to extract audio features during model training:

```bash
python main.py data/ --workspace workspace/ -O --iters 100000 --asr_model cpierse/wav2vec2-large-xlsr-53-esperanto
```

7. FFmpeg version incorrect for RTMP push streaming
Feedback from community members indicates that version 4.2.2 is required. I'm not sure which specific versions won't work. The principle is to run ffmpeg; the output information must contain libx264. If it's missing, it definitely won't work.
```
--enable-libx264
```
8. Replacing your trained model
```python
.
├── data
│   ├── data_kf.json （对应训练数据中的transforms_train.json）
│   ├── au.csv			
│   ├── pretrained
│   └── └── ngp_kf.pth （对应训练后的模型ngp_ep00xx.pth）

```

Other References
https://github.com/lipku/metahuman-stream/issues/43#issuecomment-2008930101


