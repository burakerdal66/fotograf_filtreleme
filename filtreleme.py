import gradio as gr
import cv2
import numpy as np
from matplotlib import pyplot as plt

def upload_image(image):
    height, width, _ = image.shape
    size_info = f"Width: {width} pixels, Height: {height} pixels"
    return image, size_info

def show_histogram(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.grid()
    plt.savefig('histogram.png')
    plt.close()
    return 'histogram.png'

def histogram_equalization(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    return cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def blur_image(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred

def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def median_filter(image):
    median = cv2.medianBlur(image, 5)
    return median

def gaussian_filter(image):
    gaussian = cv2.GaussianBlur(image, (15, 15), 0)
    return gaussian

def show_rgb_values(image, evt: gr.SelectData):
    x, y = int(evt.index[0]), int(evt.index[1])
    color = image[y, x]
    return color.tolist()

def apply_filter(image, x1, y1, x2, y2, filter_type):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    roi = image[y1:y2, x1:x2]
    if filter_type == "Histogram Equalization":
        roi = histogram_equalization(roi)
    elif filter_type == "Sharpen":
        roi = sharpen_image(roi)
    elif filter_type == "Blur":
        roi = blur_image(roi)
    elif filter_type == "Edge Detection":
        roi = edge_detection(roi)
    elif filter_type == "Median Filter":
        roi = median_filter(roi)
    elif filter_type == "Gaussian Filter":
        roi = gaussian_filter(roi)
    image[y1:y2, x1:x2] = roi
    return image

with gr.Blocks() as demo:
    with gr.Tab("Upload Image"):
        image_input = gr.Image(label="Upload an Image", type="numpy")
        output_image = gr.Image(label="Uploaded Image")
        image_size_text = gr.Textbox(label="Image Size", interactive=False)
        image_input.upload(upload_image, inputs=image_input, outputs=[output_image, image_size_text])

    with gr.Tab("Image Processing"):
        image_display = gr.Image(label="Image to Process")
        processing_image_size_text = gr.Textbox(label="Image Size", interactive=False)
        image_input.upload(upload_image, inputs=image_input, outputs=[image_display, processing_image_size_text])

        with gr.Row():
            histogram_btn = gr.Button("Show Histogram")
            histogram_output = gr.Image(label="Histogram")
            histogram_btn.click(show_histogram, inputs=image_input, outputs=histogram_output)

        with gr.Row():
            equalize_btn = gr.Button("Histogram Equalization")
            equalized_image = gr.Image(label="Equalized Image")
            equalize_btn.click(histogram_equalization, inputs=image_input, outputs=equalized_image)

        with gr.Row():
            sharpen_btn = gr.Button("Sharpen Image")
            sharpened_image = gr.Image(label="Sharpened Image")
            sharpen_btn.click(sharpen_image, inputs=image_input, outputs=sharpened_image)

        with gr.Row():
            blur_btn = gr.Button("Blur Image")
            blurred_image = gr.Image(label="Blurred Image")
            blur_btn.click(blur_image, inputs=image_input, outputs=blurred_image)

        with gr.Row():
            edge_btn = gr.Button("Edge Detection")
            edged_image = gr.Image(label="Edge Detected Image")
            edge_btn.click(edge_detection, inputs=image_input, outputs=edged_image)

        with gr.Row():
            median_btn = gr.Button("Median Filter")
            median_image = gr.Image(label="Median Filtered Image")
            median_btn.click(median_filter, inputs=image_input, outputs=median_image)

        with gr.Row():
            gaussian_btn = gr.Button("Gaussian Filter")
            gaussian_image = gr.Image(label="Gaussian Filtered Image")
            gaussian_btn.click(gaussian_filter, inputs=image_input, outputs=gaussian_image)

        rgb_display = gr.Textbox(label="RGB Values")
        image_display.select(show_rgb_values, inputs=image_input, outputs=rgb_display)

    with gr.Tab("Selective Area Processing"):
        selectable_image = gr.Image(label="Select Area on Image", type="numpy")
        selective_image_size_text = gr.Textbox(label="Image Size", interactive=False)
        selectable_image.upload(upload_image, inputs=selectable_image, outputs=[selectable_image, selective_image_size_text])
        x1 = gr.Number(label="X1")
        y1 = gr.Number(label="Y1")
        x2 = gr.Number(label="X2")
        y2 = gr.Number(label="Y2")
        filter_dropdown = gr.Dropdown(["Histogram Equalization", "Sharpen", "Blur", "Edge Detection", "Median Filter", "Gaussian Filter"], label="Select Filter")
        apply_btn = gr.Button("Apply Filter")
        processed_image = gr.Image(label="Processed Image")

        apply_btn.click(apply_filter, inputs=[selectable_image, x1, y1, x2, y2, filter_dropdown], outputs=processed_image)

demo.launch()
