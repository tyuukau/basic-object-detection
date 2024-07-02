import gradio as gr
from ultralytics import YOLOv10


def yolov10_inference(image, image_size, conf_threshold):
    model = YOLOv10("/content/best.pt")
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1]


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                input_type = gr.Radio(
                    choices=["Image"],
                    value="Image",
                    label="Input Type",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)

        def update_visibility(input_type):
            image = (
                gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            )
            output_image = (
                gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            )

            return image, output_image

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, output_image],
        )

        def run_inference(image, image_size, conf_threshold):
            return yolov10_inference(image, image_size, conf_threshold)

        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, image_size, conf_threshold],
            outputs=[output_image],
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Basic YOLOv10: Helmet Detection
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/tyuukau/basic-object-detection' target='_blank'>github</a>
        </h3>
        """
    )
    with gr.Row():
        with gr.Column():
            app()
if __name__ == "__main__":
    gradio_app.launch(share=True)
