import os
import gradio as gr
from demo.sam_inference import SAM_Inference
from demo.seagull_inference import Seagull
from demo.mask_utils import ImageSketcher

class Main_ui():
    def __init__(self, args) -> None: 
        self.args = args
        self.seagull = Seagull(model_path=args.model)
        
        self.example_list = self.load_example()
        self.sam = SAM_Inference()
    
    def load_example(self):
        examples = []
        for file in sorted(os.listdir(self.args.example_path)):
            examples.append([os.path.join(self.args.example_path, file)])
        return examples
            
    def load_demo(self):
        with gr.Blocks() as demo:
            preprocessed_img = gr.State(value=None)
            binary_mask = gr.State(value=None)
            
            with gr.Row():
                gr.Markdown("""
                            <img src="https://github.com/chencn2020/SEAGULL/raw/main/imgs/Logo/logo.png" alt="SEAGULL" style="height: auto; width: 100%; margin-bottom: 3%;">
                            
                            ## 🔔 Usage
                            
                            Firstly, you need to upload an image and choose the analyse types **(quality score, importance score and distortion analysis)**. 
                            
                            Then you can click **(points)** or pull a frame **(bbox)** on the image to indicate the region of interest (ROIs). 
                            
                            After that, this demo process the following steps:
                            
                            > 1. SAM extracts the mask-based ROIs based on your clicked points or frame.
                            
                            > 2. Based on the uploaded image and mask-based ROIs, SEAGULL analyses the quality of the ROIs.
                        
                            """)

            with gr.TabItem("Mask-based ROIs (Points)"):
                with gr.Row():
                    input_image_ponit = gr.Image(type="numpy", label='Input image', height=512) # input image
                    output_mask_ponit = gr.Image(label='Mask-based ROI', height=512) # output binary mask
                
                with gr.Row():
                    output_mask_point_on_img = gr.Image(label='Mask on image', height=512) # mask on image for better view
                        
                    with gr.Column():
                        radio_point = gr.Radio(label='Analysis type', choices=['Quality Score', 'Importance Score', 'Distortion Analysis'], value='Quality Score')
                        output_text_point = gr.Textbox(label='Analysis Results')
                        point_seg_button = gr.Button('Analysis')
                        
                        point_example = gr.Dataset(label='Examples', components=[input_image_ponit], samples=self.example_list)

            with gr.TabItem("Mask-based ROIs (BBox)"):
                with gr.Row():
                    input_image_BBOX = ImageSketcher(type="numpy", label='Input image', height=512)
                    output_mask_BBOX = gr.Image(label='Mask-based ROI', height=512)
                
                with gr.Row():
                    output_BBOX_mask_on_img = gr.Image(label='Mask on image', height=512)
                        
                    with gr.Column():
                        radio_BBOX = gr.Radio(label='Analysis type', choices=['Quality Score', 'Importance Score', 'Distortion Analysis'], value='Quality Score')
                        output_text_BBOX = gr.Textbox(label='ROI Quality Analysis')
                        box_seg_button = gr.Button('Generate mask and analysis')
                        box_analyse_button = gr.Button('Analysis')
                        
                        BBOX_example = gr.Dataset(label='Examples', components=[input_image_BBOX], samples=self.example_list)

            # click point
            input_image_ponit.upload(
                self.seagull.init_image,
                [input_image_ponit],
                [preprocessed_img, input_image_ponit, input_image_BBOX]
            )

            point_example.click(
                self.seagull.init_image,
                [point_example],
                [preprocessed_img, input_image_ponit, input_image_BBOX]
            )
            
            # after clicking on the image
            input_image_ponit.select(
                self.sam.img_select_point,
                [preprocessed_img],
                [input_image_ponit, output_mask_ponit, output_mask_point_on_img, binary_mask]
            ).then(
                self.seagull.seagull_predict,
                [preprocessed_img, binary_mask, radio_point],
                [output_text_point]
            )
            
            point_seg_button.click(
                self.seagull.seagull_predict,
                [preprocessed_img, binary_mask, radio_point],
                [output_text_point]
            )

            # draw frame
            input_image_BBOX.upload(
                self.seagull.init_image,
                [input_image_BBOX],
                [preprocessed_img, input_image_ponit, input_image_BBOX]
            )

            BBOX_example.click(
                self.seagull.init_image,
                [BBOX_example],
                [preprocessed_img, input_image_ponit, input_image_BBOX]
            )

            # after drawing a frame on the image
            input_image_BBOX.select(
                self.sam.gen_box_seg,
                [input_image_BBOX],
                [output_mask_BBOX, output_BBOX_mask_on_img, binary_mask]
            )
            
            box_seg_button.click(
                self.sam.gen_box_seg,
                [input_image_BBOX],
                [output_mask_BBOX, output_BBOX_mask_on_img, binary_mask]
            ).then(
                self.seagull.seagull_predict,
                [preprocessed_img, binary_mask, radio_BBOX],
                [output_text_BBOX]
            )
            
            box_analyse_button.click(
                self.seagull.seagull_predict,
                [preprocessed_img, binary_mask, radio_BBOX],
                [output_text_BBOX]
            )

        return demo