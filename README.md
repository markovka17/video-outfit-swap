# video-outfit-swap

## Requirements
1. git clone https://github.com/yisol/IDM-VTON/ and follow the desired requirements (IDM-VTON/environment.yaml).
2. Based on https://colab.research.google.com/github/camenduru/IDM-VTON-jupyter/blob/main/IDM_VTON_jupyter.ipynb:
    * https://huggingface.co/camenduru/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl;
    * https://huggingface.co/camenduru/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx;
    * https://huggingface.co/camenduru/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx;
    * https://huggingface.co/camenduru/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth.
   Unpack into IDM-VTON/ckpt.
3. All other checkpoints will be automatically downloaded by Hugging Face utils under the hood.

## Pipeline
Given a video and a reference image, we compute per-frame person's garment mask, hands mask, densepose mask and apply diffusion-based IDM-VTON for outfit swapping. To improve temporal consistency, we apply mask smoothing along space and time. To avoid artifacts, we repair the result so that we blend the original frame with the generated one in the region of the garment mask, thus maintaining the original quality of the person in the video.
For this purpose, we use DensePose, Segformer, OpenPose, Humanparsing and IDM-VTON models.

NOTE: To fix flickering, I suggest using https://github.com/antgroup/echomimic_v2.


## How to run
```bash
python inference.py 
    --video-path path-to-video
    --garment-path path-to-garment
    --end-frame N
    --output-path output.mp4
```

## Example

Source | Generated
:-: | :-:
<video src='https://github.com/user-attachments/assets/c61b4b37-7b08-4d4b-9562-0384db05af56' width=384/> | <video src='https://github.com/user-attachments/assets/da5a9bd3-d735-4a5f-af5c-9f6c7c61d3cb' width=384/>

## Failed ideas
1. I tried https://github.com/Zheng-Chong/CatVTON and got a bad result in terms of fitting artifacts.
2. I tried https://github.com/Zheng-Chong/CatV2TON and also got a bad result in terms of fitting artifacts, especially temporal consistency and poor resolution (even after applying a super resolution model).
