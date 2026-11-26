from ultralytics import YOLO

# 1. Load the YOLOv8 nano classification model
model = YOLO("yolov8n-cls.pt")

# 2. Train on your reorganized bottle dataset
model.train(
    data=r"C:\yolo_project\bottle_cls_dataset",
    epochs=30,
    imgsz=224,
    batch=8,
    project="bottle_runs",
    name="defect_cls"
)

# 3. Load best trained model and export to .pte for Raspberry Pi 5
best_model = YOLO(r"bottle_runs\defect_cls\weights\best.pt")
best_model.export(format="executorch", imgsz=224)

print("Export complete! Copy 'best_executorch_model' to your Raspberry Pi 5.")