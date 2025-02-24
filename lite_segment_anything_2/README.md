对于90帧的视频，只在第1帧添加一个prompt：

```python
## load_video_frames time:  3.273268699645996
#### init_state time:  8.88269829750061
```

load_video_frames切分90帧耗时3.27秒（约27.5fps，每帧耗时0.036秒）

image_encoder处理90帧耗时5.61秒（约16fps，每帧耗时0.06秒）

------

```python
prompt_encoder time:  0.029015779495239258
mask_decoder time:  0.0900428295135498
# _track_step time:  0.1449282169342041
Memory encoding time:  9.5367431640625e-07
## _run_single_frame_inference time:  0.14565563201904297
#### add_new_points_or_box time:  0.14637327194213867
```

prompt_encoder处理一帧耗时0.03秒（约33fps）

mask_decoder处理一帧耗时0.09秒（约11fps）

memory_encoder几乎不耗时

------

```python
memory_attn time: 0.003690958023071289
prompt_encoder time:  0.0012328624725341797
mask_decoder time:  0.0029261112213134766
# _track_step time:  0.008767366409301758
Memory encoding time:  4.76837158203125e-07
## _run_single_frame_inference time:  0.24123287200927734
#### add_new_points_or_box time:  0.2416074275970459
```

非首帧条件帧时_run_single_frame_inference除了

------

- _track_step（）函数的其他部分每条件帧耗时约1.5秒

```python
### propagate_in_video_preflight time:  0.023334741592407227
memory_attn time: 0.004694223403930664
prompt_encoder time:  0.0006735324859619141
mask_decoder time:  0.0029833316802978516
# _track_step time:  0.00987386703491211
Memory encoding time:  0.0015652179718017578
## _run_single_frame_inference time:  0.2621490955352783
memory_attn time: 0.003061056137084961
prompt_encoder time:  0.0005469322204589844
mask_decoder time:  0.0027801990509033203
# _track_step time:  0.007121086120605469
Memory encoding time:  0.001478433609008789
## _run_single_frame_inference time:  0.009647846221923828
...
memory_attn time: 0.0027790069580078125
prompt_encoder time:  0.0004360675811767578
mask_decoder time:  0.0022406578063964844
# _track_step time:  0.006135463714599609
Memory encoding time:  0.0011966228485107422
## _run_single_frame_inference time:  0.007917404174804688
#### propagate_in_video time:  1.2778618335723877
```

memory_attn实际耗时约为mask_decoder的1.1-1.5倍，每一帧约为0.003-0.0045秒

mask_decoder处理每一帧用时约为0.002-0.003秒

prompt_encoder在非条件帧中几乎不耗时

memory_encoder处理每一帧用时约为0.001-0.0015秒

_track_step（）函数中对于非条件帧，每一帧耗时约为0.007秒

_run_single_frame_inference time（）函数包含 _track_step（）函数和 memory_encoder 模块，非条件帧耗时越0.008秒

------

综上总结：

尽量减少条件帧占比，条件帧耗时是非条件帧耗时的200倍

在条件帧中，添加条件帧提示中的_run_single_frame_inference（）函数非_track_step（）部分耗时约0.2秒，此处**需要重点优化**！



对于神经网络的模块部分：

memory_encoder几乎不耗时，非条件帧中prompt_encoder几乎不耗时

耗时顺序大致为：

image_encoder（每帧0.06s）>

mask_decoder（每条件帧0.09s/每非条件帧0.003s）= 

memory_attn（每条件帧0.004s/每非条件帧0.005s）> 

memory_encoder（每条件帧0s/每非条件帧0.001s）=

prompt_encoder（每条件帧0.001s/每非条件帧0s）