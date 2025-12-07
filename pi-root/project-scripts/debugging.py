
def debug(os, cv2, preprocess, session, parse_output):
    print("Begin debugging script", flush=True)
    print("cwd =", os.getcwd(), flush=True)
    frame = cv2.imread("/home/danbitter/yolov5-6.1/pi-root/project-scripts/freight_train.jpg")
    print("frame is None?", frame is None, flush=True)

    img = preprocess(frame)

    # input_name must match your model’s input tensor name. You can print it using:
    input_name = session.get_inputs()[0].name
    # print("input_name to match model input tensor name", input_name, flush=True) # images
    outputs = session.run(None, {input_name: img})

    # # Print the raw ONNX output shape(s)
    # print("Number of outputs:", len(outputs), flush=True)
    # for i, out in enumerate(outputs):
    #     print(f"Output[{i}] shape:", out.shape, flush=True)
    #     # For safety, only print the first few rows
    #     print(f"Output[{i}] sample:", out.reshape(-1, out.shape[-1])[:5], flush=True)

    conf, class_ids = parse_output(outputs)
    print("parse_output: ", conf, class_ids, flush=True)

import onnx

model = onnx.load("best.onnx")

# Print all metadata entries (if YOLOv8 exported them correctly)
print("Model metadata:")
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")
